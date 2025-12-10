import argparse
import os
import pdb
import pickle
import random
import shutil
import time
from copy import deepcopy
import torch.nn.init as init

import arg_parser
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from trainer import train, validate
from utils import *
from utils import NormalizeByChannelMeanStd

best_sa = 0


def main():
    global args, best_sa
    args = arg_parser.parse_args()

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    args.class_to_replace = None  # safe default for any dataset

    result = setup_model_dataset(args)

    if args.dataset == "imagenet":
        model, train_loader, val_loader = result

    elif args.dataset == "imagenet_zeus":
        model, retain_loader, forget_loader, val_loader = result
        train_loader = retain_loader

    else:
        model, train_loader, val_loader, test_loader, marked_loader = result


    model.cuda()
    
    # === Optional weight reinitialization (only for training from scratch) ===
    # NOTE: Comment out the next section if you want to use pretrained weights
    """
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # Only reinitialize weights if NOT using pretrained model
    if not getattr(args, "pretrained", False):
        model.apply(weights_init)
        print("[DEBUG] Model weights reinitialized.")
    else:
        print("[DEBUG] Skipped weight reinit: using pretrained weights.")
    """
    
    # For TinyImageNet, we should use pretrained ImageNet features
    if hasattr(args, 'pretrained') and args.pretrained:
        print("[DEBUG] Using pretrained ImageNet features (weights NOT reinitialized)")
    else:
        print("[DEBUG] Training from scratch (weights randomly initialized)")


    print("\n[DEBUG] Model Summary:")
    print(model)
    print(f"[DEBUG] fc input: {model.fc.in_features}, output: {model.fc.out_features}\n")


    print(f"number of train dataset {len(train_loader.dataset)}")
    print(f"number of val dataset {len(val_loader.dataset)}")

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.imagenet_arch:
        lambda0 = (
            lambda cur_iter: (cur_iter + 1) / args.warmup
            if cur_iter < args.warmup
            else (
                0.5
                * (
                    1.0
                    + np.cos(
                        np.pi * ((cur_iter - args.warmup) / (args.epochs - args.warmup))
                    )
                )
            )
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=decreasing_lr, gamma=0.1
        )  # 0.1 is fixed
    if args.resume:
        print("resume from checkpoint {}".format(args.checkpoint))
        checkpoint = torch.load(
            args.checkpoint, map_location=torch.device("cuda:" + str(args.gpu))
        )
        best_sa = checkpoint["best_sa"]
        start_epoch = checkpoint["epoch"]
        all_result = checkpoint["result"]

        model.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        initalization = checkpoint["init_weight"]
        print("loading from epoch: ", start_epoch, "best_sa=", best_sa)

    else:
        all_result = {}
        all_result["train_ta"] = []
        all_result["test_ta"] = []
        all_result["val_ta"] = []

        start_epoch = 0
        state = 0

    print(f"[DEBUG] model.fc in_features: {model.fc.in_features}")
    print(f"[DEBUG] model.fc out_features: {model.fc.out_features}")


    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        print(
            "Epoch #{}, Learning rate: {}".format(
                epoch, optimizer.state_dict()["param_groups"][0]["lr"]
            )
        )
        acc = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        tacc = validate(val_loader, model, criterion, args)
        scheduler.step()

        all_result["train_ta"].append(acc)
        all_result["val_ta"].append(tacc)

        # remember best prec@1 and save checkpoint
        is_best_sa = tacc > best_sa
        best_sa = max(tacc, best_sa)

        save_checkpoint(
            {
                "result": all_result,
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_sa": best_sa,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_SA_best=is_best_sa,
            pruning=state,
            save_path=args.save_dir,
        )
        # Also save the model as 'original_model.pth' for generate_mask.py


        ckpt_path = os.path.join(args.save_dir, "checkpoint.pth.tar")
        orig_path = os.path.join(args.save_dir, "original_model.pth")

        if os.path.exists(ckpt_path):
            shutil.copyfile(ckpt_path, orig_path)

        print("one epoch duration:{}".format(time.time() - start_time))

    # plot training curve
    plt.plot(all_result["train_ta"], label="train_acc")
    plt.plot(all_result["val_ta"], label="val_acc")
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, str(state) + "net_train.png"))
    plt.close()

    plt.savefig(os.path.join(args.save_dir, "thesis_inceptionv3_plot.png"))


    with open(os.path.join(args.save_dir, "thesis_metrics.txt"), "w") as f:
        f.write(f"Best validation accuracy: {best_sa:.4f}\n")
        f.write(f"Train samples: {len(train_loader.dataset)}\n")
        f.write(f"Val samples: {len(val_loader.dataset)}\n")
        f.write("Train Accuracy Curve:\n")
        f.write(", ".join([f"{x:.4f}" for x in all_result["train_ta"]]) + "\n")
        f.write("Val Accuracy Curve:\n")
        f.write(", ".join([f"{x:.4f}" for x in all_result["val_ta"]]) + "\n")


    print("Performance on the test data set")
    test_tacc = validate(val_loader, model, criterion, args)
    if len(all_result["val_ta"]) != 0:
        val_pick_best_epoch = np.argmax(np.array(all_result["val_ta"]))
        print(
            "* best SA = {}, Epoch = {}".format(
                all_result["val_ta"][val_pick_best_epoch], val_pick_best_epoch + 1
            )
        )
    

if __name__ == "__main__":
    main()