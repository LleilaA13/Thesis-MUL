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

def restore_flipped_forget_labels(loader):
    # If the dataset is a Subset, go deeper
    dataset = loader.dataset
    if hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    if hasattr(dataset, "targets"):
        for i in range(len(dataset.targets)):
            if dataset.targets[i] < 0:
                dataset.targets[i] = -dataset.targets[i] - 1

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
    # prepare dataset
 
    if args.dataset == "imagenet_zeus":
        model, retain_loader, forget_loader, val_loader = utils.setup_model_dataset(args)
        test_loader = val_loader  # reuse val_loader as test_loader
        marked_loader = None      # not used
    else:
        model, train_loader_full, val_loader, test_loader, marked_loader = utils.setup_model_dataset(args)

    model.cuda()

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=12,
            pin_memory=True,
            shuffle=shuffle,
        )

    if args.dataset == "imagenet_zeus":
        from torch.utils.data import Subset

        dataset = retain_loader.dataset.dataset if isinstance(retain_loader.dataset, Subset) else retain_loader.dataset
        forget_mask = torch.load(args.subset_indices_path).bool()

        print("[DEBUG] Loaded forget mask of length:", len(forget_mask))
        print("[DEBUG] #Forget samples:", forget_mask.sum().item())
        print("[DEBUG] #Retain samples:", (~forget_mask).sum().item())

        forget_ids = [i for i, flag in enumerate(forget_mask) if flag]
        retain_ids = [i for i, flag in enumerate(forget_mask) if not flag]

        for i in forget_ids:
            dataset.targets[i] = -dataset.targets[i] - 1

        forget_dataset = Subset(dataset, forget_ids)
        retain_dataset = Subset(dataset, retain_ids)

        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)

        ...

    elif args.dataset == "svhn":
        forget_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = forget_dataset.targets < 0
        except:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )
    else:
        from torch.utils.data import Subset

        # Load full dataset and the forget mask
        if isinstance(marked_loader.dataset, Subset):
          dataset = marked_loader.dataset.dataset
        else:
         dataset = marked_loader.dataset

        forget_mask = torch.load(args.subset_indices_path).bool()  # 1 = forget, 0 = retain

        print("[DEBUG] Loaded forget mask of length:", len(forget_mask))
        print("[DEBUG] #Forget samples:", forget_mask.sum().item())
        print("[DEBUG] #Retain samples:", (~forget_mask).sum().item())

        # Construct forget and retain indices
        all_indices = list(range(len(forget_mask)))
        forget_ids = [i for i, flag in enumerate(forget_mask) if flag]
        retain_ids = [i for i, flag in enumerate(forget_mask) if not flag]

        # Optionally apply SalUn convention to forget labels
        for i in forget_ids:
            dataset.targets[i] = -dataset.targets[i] - 1


        # Create subset datasets
        forget_dataset = Subset(dataset, forget_ids)
        retain_dataset = Subset(dataset, retain_ids)

        # Wrap in DataLoaders
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)

 

    print(f"number of retain dataset {len(retain_dataset)}")
    print(f"number of forget dataset {len(forget_dataset)}")
    
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = nn.CrossEntropyLoss()

    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:

        checkpoint = torch.load(args.model_path, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        if args.mask_path:
            mask = torch.load(args.mask_path)

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
                restore_flipped_forget_labels(loader)  
            if name == "val":
                print("[!] Skipping 'val' loader due to previous crash.")
                continue

            val_acc = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")

        print("Evaluating on true forget/retain test sets...")

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

        forget_test_loader = DataLoader(Subset(val_dataset, forget_test_ids), batch_size=args.batch_size, shuffle=False)
        retain_test_loader = DataLoader(Subset(val_dataset, retain_test_ids), batch_size=args.batch_size, shuffle=False)

        forget_test_acc = validate(forget_test_loader, model, criterion, args)
        retain_test_acc = validate(retain_test_loader, model, criterion, args)

        accuracy["forget_test"] = forget_test_acc
        accuracy["retain_test"] = retain_test_acc

        print(f"forget_test acc: {forget_test_acc}")
        print(f"retain_test acc: {retain_test_acc}")  

        print("\n[*] Evaluating on true forget/retain training sets...")

        # Get access to original training dataset
        train_dataset = retain_loader.dataset.dataset if hasattr(retain_loader.dataset, "dataset") else retain_loader.dataset

        forget_train_ids = [i for i, (_, label) in enumerate(train_dataset.samples) if label in CAT_CLASS_IDS]
        retain_train_ids = [i for i, (_, label) in enumerate(train_dataset.samples) if label not in CAT_CLASS_IDS]

        forget_train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_dataset, forget_train_ids),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )

        retain_train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_dataset, retain_train_ids),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )

        forget_train_acc = validate(forget_train_loader, model, criterion, args)
        retain_train_acc = validate(retain_train_loader, model, criterion, args)

        print(f"[Train Forget Accuracy]: {forget_train_acc:.2f}%")
        print(f"[Train Retain Accuracy]: {retain_train_acc:.2f}%")

        import json

        results = {
            "train_forget_accuracy": forget_train_acc,
            "train_retain_accuracy": retain_train_acc,
            "test_forget_accuracy": forget_test_acc,
            "test_retain_accuracy": retain_test_acc
        }

        with open(f"{args.save_dir}/salun_eval_results.json", "w") as f:
            json.dump(results, f, indent=2)

        
        evaluation_result["new_accuracy"] = accuracy  
        evaluation_result["accuracy"] = accuracy
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        if deprecated in evaluation_result:
            evaluation_result.pop(deprecated)

    """forget efficacy MIA:
        in distribution: retain
        out of distribution: test
        target: (, forget)"""
    if "SVC_MIA_forget_efficacy" not in evaluation_result:
        test_len = len(test_loader.dataset)
        forget_len = len(forget_dataset)
        retain_len = len(retain_dataset)

        utils.dataset_convert_to_test(retain_dataset, args)
        utils.dataset_convert_to_test(forget_loader, args)
        utils.dataset_convert_to_test(test_loader, args)

        shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
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
