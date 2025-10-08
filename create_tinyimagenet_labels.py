#!/usr/bin/env python3
"""
Generate correct label files for TinyImageNet
The existing label files are for ImageNet-1K (1000 classes) but we need TinyImageNet (200 classes)
"""

import torch
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def create_tinyimagenet_labels():
    """Create correct train_ys.pth and val_ys.pth for TinyImageNet"""
    
    print("🔧 CREATING CORRECT TINYIMAGENET LABELS")
    print("=" * 50)
    
    # Paths
    train_dir = "datasets/tiny-imagenet-200/train"
    val_dir = "datasets/tiny-imagenet-200/val"
    
    if not os.path.exists(train_dir):
        print(f"❌ Training directory not found: {train_dir}")
        return False
        
    if not os.path.exists(val_dir):
        print(f"❌ Validation directory not found: {val_dir}")
        return False
    
    # Create training labels
    print("📁 Processing training set...")
    train_dataset = ImageFolder(train_dir)
    print(f"   Classes found: {len(train_dataset.classes)}")
    print(f"   Samples found: {len(train_dataset)}")
    
    # Extract labels in the order of the dataset
    train_labels = []
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        train_labels.append(label)
    
    train_labels_tensor = torch.tensor(train_labels)
    
    # Create validation labels 
    print("📁 Processing validation set...")
    val_dataset = ImageFolder(val_dir)
    print(f"   Classes found: {len(val_dataset.classes)}")
    print(f"   Samples found: {len(val_dataset)}")
    
    # Extract labels in the order of the dataset
    val_labels = []
    for i in range(len(val_dataset)):
        _, label = val_dataset[i]
        val_labels.append(label)
        
    val_labels_tensor = torch.tensor(val_labels)
    
    # Verify the labels
    print("\n🔍 VERIFICATION:")
    print(f"Training labels shape: {train_labels_tensor.shape}")
    print(f"Training unique classes: {len(torch.unique(train_labels_tensor))}")
    print(f"Training label range: {train_labels_tensor.min().item()} to {train_labels_tensor.max().item()}")
    
    print(f"Validation labels shape: {val_labels_tensor.shape}")
    print(f"Validation unique classes: {len(torch.unique(val_labels_tensor))}")
    print(f"Validation label range: {val_labels_tensor.min().item()} to {val_labels_tensor.max().item()}")
    
    # Check for dog classes
    dog_indices = [11, 39, 78, 135, 182, 194]
    train_dogs = [idx for idx in dog_indices if idx in torch.unique(train_labels_tensor)]
    val_dogs = [idx for idx in dog_indices if idx in torch.unique(val_labels_tensor)]
    print(f"Dog classes in training: {train_dogs}")
    print(f"Dog classes in validation: {val_dogs}")
    
    # Save the correct labels
    os.makedirs("labels_tinyimagenet", exist_ok=True)
    
    train_path = "labels_tinyimagenet/train_ys.pth"
    val_path = "labels_tinyimagenet/val_ys.pth"
    
    torch.save(train_labels_tensor, train_path)
    torch.save(val_labels_tensor, val_path)
    
    print(f"\n✅ SAVED CORRECT LABELS:")
    print(f"   Training: {train_path}")
    print(f"   Validation: {val_path}")
    
    # Show class mapping for dog classes
    print(f"\n🐕 DOG CLASS MAPPING:")
    with open('datasets/tiny-imagenet-200/wnids.txt', 'r') as f:
        wnids = [line.strip() for line in f.readlines()]
    
    word_map = {}
    with open('datasets/tiny-imagenet-200/words.txt', 'r') as f:
        for line in f:
            if '\t' in line:
                wnid, words = line.strip().split('\t', 1)
                word_map[wnid] = words
    
    for idx in dog_indices:
        if idx < len(wnids):
            wnid = wnids[idx]
            description = word_map.get(wnid, 'Unknown')
            print(f"   Index {idx:3d}: {wnid} - {description}")
    
    return True

if __name__ == "__main__":
    success = create_tinyimagenet_labels()
    if success:
        print("\n🎯 UPDATE YOUR SCRIPT:")
        print("Change the label paths in resnet50_unlearn_dogs.py to:")
        print("  --train_y_file labels_tinyimagenet/train_ys.pth")
        print("  --val_y_file labels_tinyimagenet/val_ys.pth")
    else:
        print("\n❌ Failed to create labels. Check TinyImageNet dataset structure.")