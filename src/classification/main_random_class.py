import copy
import os
from collections import OrderedDict

import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils
from trainer import validate

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from PIL import Image

def restore_flipped_forget_labels(loader):
    """Restore original labels that were flipped during Random Labels training"""
    dataset = loader.dataset
    if hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    if hasattr(dataset, "targets"):
        # Check for original forget labels (random labels approach)
        if hasattr(dataset, "original_forget_labels"):
            original_labels = dataset.original_forget_labels
            restored_count = 0
            for i, original_label in original_labels.items():
                dataset.targets[i] = original_label
                restored_count += 1
            print(f"[DEBUG] Restored {restored_count} random labels to original labels")
            return
            
        # Fallback: Original method for negative labels (if used)
        original_count = 0
        for i in range(len(dataset.targets)):
            if dataset.targets[i] < 0:
                dataset.targets[i] = -dataset.targets[i] - 1
                original_count += 1
        if original_count > 0:
            print(f"[DEBUG] Restored {original_count} negative labels to original labels")
        else:
            print(f"[DEBUG] No negative labels found - labels may need manual restoration for RL method")

def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed

    if args.dataset == "imagenet_zeus":
        model, retain_loader, forget_loader, val_loader = utils.setup_model_dataset(args)
        dataset = retain_loader.dataset.dataset if isinstance(retain_loader.dataset, Subset) else retain_loader.dataset
        test_loader = val_loader
        marked_loader = None
        retain_dataset = retain_loader.dataset

    else:
        model, train_loader_full, val_loader, test_loader, marked_loader = utils.setup_model_dataset(args)
        
        # Create retain_loader and forget_loader from marked_loader        
        def replace_loader_dataset(dataset, batch_size=args.batch_size, seed=1, shuffle=True):
            utils.setup_seed(seed)
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=0,
                pin_memory=True,
                shuffle=shuffle,
            )
        
        # Load full dataset and the forget mask
        if isinstance(marked_loader.dataset, Subset):
            dataset = marked_loader.dataset.dataset
        else:
            dataset = marked_loader.dataset

        forget_mask = torch.load(args.subset_indices_path).bool()  # 1 = forget, 0 = retain

        # Construct forget and retain indices
        forget_ids = [i for i, flag in enumerate(forget_mask) if flag]
        retain_ids = [i for i, flag in enumerate(forget_mask) if not flag]

        # Note: Random labels will be assigned by RL.py unlearn method
        # We just create the subset datasets here

        # Create subset datasets preserving original transforms
        forget_dataset = Subset(dataset, forget_ids)
        retain_dataset = Subset(dataset, retain_ids)

        # Get the original transform from the dataset
        original_transform = getattr(dataset, 'transform', None)
        print(f"[DEBUG] Original dataset transform: {original_transform}")
        
        # Check if the transform already has ToTensor and normalization
        if hasattr(dataset, 'transform') and dataset.transform is not None:
            from torchvision import transforms
            
            if isinstance(original_transform, transforms.Compose):
                transform_list = list(original_transform.transforms)
            else:
                transform_list = [original_transform]
            
            # Check if ToTensor and Normalize are already present
            has_to_tensor = any(isinstance(t, transforms.ToTensor) for t in transform_list)
            has_normalize = any(isinstance(t, transforms.Normalize) for t in transform_list)
            
            # Only add missing transforms
            if not has_to_tensor:
                transform_list.append(transforms.ToTensor())
            if not has_normalize:
                transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            
            # Update the transform only if we added something
            if not has_to_tensor or not has_normalize:
                complete_transform = transforms.Compose(transform_list)
                dataset.transform = complete_transform
                print(f"[DEBUG] Updated transform: {complete_transform}")
            else:
                print(f"[DEBUG] Transform already complete: {original_transform}")
        else:
            # If no transform, we need to set a proper one for TinyImageNet
            from torchvision import transforms
            tinyimagenet_transform = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            dataset.transform = tinyimagenet_transform
            print(f"[DEBUG] Set TinyImageNet transform: {tinyimagenet_transform}")

        # Wrap in DataLoaders
        forget_loader = replace_loader_dataset(forget_dataset, seed=args.seed, shuffle=True)
        retain_loader = replace_loader_dataset(retain_dataset, seed=args.seed, shuffle=True)

    model.cuda()

    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = nn.CrossEntropyLoss()
    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)
        if checkpoint is not None:
            model, evaluation_result = checkpoint
        if evaluation_result is None or not isinstance(evaluation_result, dict):
            evaluation_result = {}
            # ✅ Skip if accuracy already exists
            if "accuracy" in evaluation_result:
                print("[✓] Skipping evaluation — already in checkpoint")
                evaluation_result["new_accuracy"] = True
        
    else:
        checkpoint = torch.load(args.model_path, map_location=device)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        mask = torch.load(args.mask_path) if args.mask_path else None

        if args.unlearn != "retrain":
            model.load_state_dict(checkpoint, strict=False)

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        unlearn_method(unlearn_data_loaders, model, criterion, args, mask)
        unlearn.save_unlearn_checkpoint(model, None, args)

    if evaluation_result is None:
        evaluation_result = {}
    if "new_accuracy" not in evaluation_result:
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            utils.dataset_convert_to_test(loader.dataset, args)
            if name == "forget":
                # For forget evaluation: restore original labels to test how poorly 
                # the model performs on what it was trained to forget
                print(f"[DEBUG] Restoring original labels for proper forget evaluation")
                restore_flipped_forget_labels(loader)

            # Patch transform directly before validation
            if args.dataset == "TinyImagenet":
                resize_transform = transforms.Compose([
                    transforms.Resize(64),  # TinyImageNet is 64x64
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])
            else:
                # ImageNet/ImageNet Zeus transform
                resize_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])
            target_dataset = loader.dataset
            while hasattr(target_dataset, "dataset"):
                target_dataset = target_dataset.dataset
            target_dataset.transform = resize_transform

            val_acc = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")

        print("Evaluating on true forget/retain test sets...")
        
        if args.dataset == "imagenet_zeus":
            # Original ImageNet Zeus evaluation
            val_dir = "/media/pinas/datasets/imagenet_zeus/val"
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            val_dataset = ImageFolder(val_dir, transform=val_transform)
            CAT_CLASS_IDS = [281, 282, 283, 284, 285]
            forget_test_ids = [i for i, (_, label) in enumerate(val_dataset.samples) if label in CAT_CLASS_IDS]
            retain_test_ids = [i for i, (_, label) in enumerate(val_dataset.samples) if label not in CAT_CLASS_IDS]
        
        elif args.dataset == "TinyImagenet":
            # TinyImageNet evaluation using centralized config
            import sys
            sys.path.append('/media/hdd/usr/leyla/Unlearn-Saliency')
            from unlearn_config import get_forget_class_config
            
            val_dir = "datasets/tiny-imagenet-200/val"
            val_transform = transforms.Compose([
                transforms.Resize(64),  # TinyImageNet is 64x64
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            
            # Load TinyImageNet validation dataset using the helper class first
            from dataset import TinyImageNetImageFolder, TinyImageNetDataset
            val_image_folder = TinyImageNetImageFolder(val_dir, transform=val_transform)
            val_dataset = TinyImageNetDataset(val_image_folder)
            
            # Auto-detect forget type from mask path or use default
            forget_type = 'dogs'  # Default
            if hasattr(args, 'subset_indices_path') and args.subset_indices_path:
                if 'cat' in args.subset_indices_path.lower():
                    forget_type = 'cats'
                elif 'vehicle' in args.subset_indices_path.lower():
                    forget_type = 'vehicles'
                elif 'dog' in args.subset_indices_path.lower():
                    forget_type = 'dogs'
            
            # Get configuration for the detected forget type
            config = get_forget_class_config(forget_type)
            FORGET_CLASS_IDS = config['indices']
            
            print(f"Detected forget type: {forget_type}")
            print(f"Found {len(FORGET_CLASS_IDS)} {forget_type} classes in validation set: {FORGET_CLASS_IDS}")
            print(f"Class names: {config['names']}")
            print(f"Total validation samples: {len(val_dataset.imgs)}")
            
            forget_test_ids = [i for i, (_, label) in enumerate(val_dataset.imgs) if label in FORGET_CLASS_IDS]
            retain_test_ids = [i for i, (_, label) in enumerate(val_dataset.imgs) if label not in FORGET_CLASS_IDS]
            
            print(f"Forget test samples: {len(forget_test_ids)}")
            print(f"Retain test samples: {len(retain_test_ids)}")
            
            # Create subset datasets that preserve transforms
            class TransformSubset(torch.utils.data.Dataset):
                def __init__(self, dataset, indices, transform):
                    self.dataset = dataset
                    self.indices = indices
                    self.transform = transform
                    
                def __getitem__(self, idx):
                    img_path, target = self.dataset.imgs[self.indices[idx]]
                    img = Image.open(img_path).convert("RGB")
                    if self.transform is not None:
                        img = self.transform(img)
                    return img, target
                    
                def __len__(self):
                    return len(self.indices)
            
            forget_test_dataset = TransformSubset(val_dataset, forget_test_ids, val_transform)
            retain_test_dataset = TransformSubset(val_dataset, retain_test_ids, val_transform)
            
        else:
            # Skip true test evaluation for other datasets
            print(f"True test evaluation not implemented for dataset: {args.dataset}")
            accuracy["forget_test"] = 0.0
            accuracy["retain_test"] = 0.0
            evaluation_result["accuracy"] = accuracy
            unlearn.save_unlearn_checkpoint(model, evaluation_result, args)
            return

        forget_test_loader = DataLoader(forget_test_dataset, batch_size=args.batch_size, shuffle=False)
        retain_test_loader = DataLoader(retain_test_dataset, batch_size=args.batch_size, shuffle=False)

        forget_test_acc = validate(forget_test_loader, model, criterion, args)
        retain_test_acc = validate(retain_test_loader, model, criterion, args)

        accuracy["forget_test"] = forget_test_acc
        accuracy["retain_test"] = retain_test_acc

        print(f"forget_test acc: {forget_test_acc}")
        print(f"retain_test acc: {retain_test_acc}")

        evaluation_result["accuracy"] = accuracy
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        evaluation_result.pop(deprecated, None)

    if "SVC_MIA_forget_efficacy" not in evaluation_result:
        test_len = len(test_loader.dataset)
        forget_len = len(forget_loader.dataset)
        retain_len = len(retain_loader.dataset)

        utils.dataset_convert_to_test(retain_dataset, args)
        utils.dataset_convert_to_test(forget_loader, args)
        utils.dataset_convert_to_test(test_loader, args)


        shadow_train = torch.utils.data.Subset(retain_loader.dataset, list(range(test_len)))

        
        if args.dataset == "TinyImagenet":
            resize_transform = transforms.Compose([
                transforms.Resize(64),  # TinyImageNet is 64x64
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            # ImageNet/ImageNet Zeus transform
            resize_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        for dset in [shadow_train.dataset, test_loader.dataset, forget_loader.dataset]:
            target_dataset = dset
            while hasattr(target_dataset, "dataset"):
                target_dataset = target_dataset.dataset
            target_dataset.transform = resize_transform
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False
        )

        evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=test_loader,
            target_train=None,
            target_test=forget_loader,
            model=model,
        )
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

if __name__ == "__main__":
    main()
