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

    if args.dataset == "imagenet_zeus":
        model, retain_loader, forget_loader, val_loader = utils.setup_model_dataset(args)
        dataset = retain_loader.dataset.dataset if isinstance(retain_loader.dataset, Subset) else retain_loader.dataset
        test_loader = val_loader
        marked_loader = None
        retain_dataset = retain_loader.dataset

    else:
        model, train_loader_full, val_loader, test_loader, marked_loader = utils.setup_model_dataset(args)

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
                restore_flipped_forget_labels(loader)

            # Patch transform directly before validation
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
