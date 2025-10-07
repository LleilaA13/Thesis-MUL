#!/usr/bin/env python3
"""
Debug script to check TinyImageNet data structure
"""

import os
import sys
from torchvision.datasets import ImageFolder
from torchvision import transforms

# Add the src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "src", "classification"))

# Test ImageFolder directly
train_path = "/media/hdd/usr/leyla/Unlearn-Saliency/datasets/tiny-imagenet-200/train/"
print(f"Checking ImageFolder on: {train_path}")

try:
    # Simple transform for testing
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Try to create ImageFolder
    dataset = ImageFolder(train_path, transform=transform)
    print(f"Success! Found {len(dataset)} images in {len(dataset.classes)} classes")
    print(f"First few classes: {dataset.classes[:5]}")
    print(f"Sample file: {dataset.imgs[0] if len(dataset.imgs) > 0 else 'No images found'}")
    
except Exception as e:
    print(f"Error creating ImageFolder: {e}")
    
    # Check the structure manually
    print("\nChecking directory structure manually:")
    for class_dir in os.listdir(train_path)[:5]:  # Check first 5 classes
        class_path = os.path.join(train_path, class_dir)
        if os.path.isdir(class_path):
            print(f"Class {class_dir}:")
            contents = os.listdir(class_path)
            print(f"  Contents: {contents}")
            
            # Check if images are in subdirectory
            if 'images' in contents:
                images_path = os.path.join(class_path, 'images')
                image_files = [f for f in os.listdir(images_path) if f.endswith(('.JPEG', '.jpg', '.png'))]
                print(f"  Images in {class_dir}/images/: {len(image_files)} files")
            else:
                image_files = [f for f in contents if f.endswith(('.JPEG', '.jpg', '.png'))]
                print(f"  Images directly in {class_dir}/: {len(image_files)} files")