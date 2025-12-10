import time
from copy import deepcopy

import numpy as np
import torch
import utils

from .impl import iterative_unlearn


@iterative_unlearn
def RL(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    forget_dataset = deepcopy(forget_loader.dataset)
    
    if args.dataset == "cifar100" or args.dataset == "TinyImagenet":
        # Handle Subset datasets correctly
        if hasattr(forget_dataset, 'indices'):
            # This is a Subset dataset - we need to modify only the subset's targets
            base_dataset = forget_dataset.dataset
            if not hasattr(base_dataset, 'targets'):
                raise ValueError("Base dataset has no targets attribute")
            
            # Store original targets if not already stored
            if not hasattr(base_dataset, 'original_forget_labels'):
                base_dataset.original_forget_labels = {}
            
            # Create random labels only for the forget indices
            forget_indices = forget_dataset.indices
            for idx in forget_indices:
                # Store original label if not already stored
                if idx not in base_dataset.original_forget_labels:
                    base_dataset.original_forget_labels[idx] = base_dataset.targets[idx]
                # Assign random label (different from original)
                original_label = base_dataset.original_forget_labels[idx]
                random_label = np.random.randint(0, args.num_classes)
                while random_label == original_label:  # Ensure it's different
                    random_label = np.random.randint(0, args.num_classes)
                base_dataset.targets[idx] = random_label
            print(f"[DEBUG] Assigned random labels to {len(forget_indices)} forget samples")
        else:
            # Regular dataset - assign random labels normally
            try:
                forget_dataset.targets = np.random.randint(0, args.num_classes, forget_dataset.targets.shape)
            except:
                forget_dataset.dataset.targets = np.random.randint(0, args.num_classes, len(forget_dataset.dataset.targets))
    
        retain_dataset = retain_loader.dataset
        train_dataset = torch.utils.data.ConcatDataset([forget_dataset,retain_dataset])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
      
        # switch to train mode
        model.train()
      
        start = time.time()
        loader_len = len(forget_loader) + len(retain_loader)
      
        if epoch < args.warmup:
            utils.warmup_lr(epoch, i+1, optimizer,
                            one_epoch_step=loader_len, args=args)
      
        for it, (image, target) in enumerate(train_loader):
            i = it + len(forget_loader)
            image = image.cuda()
            target = target.cuda()
            output_clean = model(image)

            loss = criterion(output_clean, target)
      
            optimizer.zero_grad()
            loss.backward()
            
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            
            optimizer.step()
      
            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
      
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))
      
            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Time {3:.2f}'.format(
                          epoch, i, loader_len, end-start, loss=losses, top1=top1))
                start = time.time()
      
    elif args.dataset == "cifar10" or args.dataset == "svhn":
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
      
        # switch to train mode
        model.train()
      
        start = time.time()
        loader_len = len(forget_loader) + len(retain_loader)
      
        if epoch < args.warmup:
            utils.warmup_lr(epoch, i+1, optimizer,
                            one_epoch_step=loader_len, args=args)
        
        for i, (image, target) in enumerate(forget_loader):
            image = image.cuda()
            target = torch.randint(0, args.num_classes, target.shape).cuda()
            
            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)
            
            optimizer.zero_grad()
            loss.backward()
            
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            
            optimizer.step()
            
        for i, (image, target) in enumerate(retain_loader):
            image = image.cuda()
            target = target.cuda()
            
            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)
            
            optimizer.zero_grad()
            loss.backward()
            
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            
            optimizer.step()
            
            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))
            
            if (i + 1) % args.print_freq == 0:
               end = time.time()
               print('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Time {3:.2f}'.format(
                         epoch, i, loader_len, end-start, loss=losses, top1=top1))
               start = time.time()

        return top1.avg
    elif args.dataset == "imagenet_zeus":
        print("[INFO] RL unlearning for imagenet_zeus")
        
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()

        # Modify forget targets randomly
        forget_dataset = deepcopy(forget_loader.dataset)
        num_classes = args.num_classes if hasattr(args, "num_classes") else 1000
        try:
            forget_dataset.targets = torch.randint(0, num_classes, (len(forget_dataset),))
        except:
            forget_dataset.dataset.targets = torch.randint(0, num_classes, (len(forget_dataset.dataset.targets),))

        retain_dataset = retain_loader.dataset
        train_dataset = torch.utils.data.ConcatDataset([forget_dataset, retain_dataset])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        model.train()
        start = time.time()

        loader_len = len(train_loader)

        for i, (images, targets) in enumerate(train_loader):
            if epoch < args.warmup:
                utils.warmup_lr(epoch, i+1, optimizer, one_epoch_step=loader_len, args=args)

            images = images.cuda()
            targets = targets.cuda()

            output = model(images)
            loss = criterion(output, targets)

            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

            # metrics
            prec1 = utils.accuracy(output.data, targets)[0]
            losses.update(loss.item(), images.size(0))
            top1.update(prec1.item(), images.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Time {3:.2f}'.format(
                        epoch, i, loader_len, end-start, loss=losses, top1=top1))
                start = time.time()
        return top1.avg
