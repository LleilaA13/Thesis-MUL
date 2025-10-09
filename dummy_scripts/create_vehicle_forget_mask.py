#!/usr/bin/env python3
"""
Helper script to create vehicles_forget_indices.pt
This script identifies all samples belonging to vehicle classes in TinyImageNet
and creates a binary mask for forgetting these samples.
"""

import os
import torch
from collections import defaultdict

def create_vehicle_forget_mask():
    # Vehicle class WordNet IDs from TinyImageNet
    vehicle_wnids = [
        "n02690373",  # airliner
        "n02958343",  # car wheel
        "n02974003",  # car mirror
        "n03100240",  # convertible
        "n03417042",  # garbage truck
        "n03770679",  # minivan
        "n03796401",  # moving van
        "n03930630",  # pickup truck
        "n04037443",  # racer
        "n04285008",  # sports car
        "n04461696",  # trailer truck
    ]
    
    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tiny_imagenet_dir = os.path.join(current_dir, "datasets", "tiny-imagenet-200")
    train_dir = os.path.join(tiny_imagenet_dir, "train")
    
    # Read class mappings
    wnid_to_idx = {}
    with open(os.path.join(tiny_imagenet_dir, "wnids.txt"), 'r') as f:
        for idx, line in enumerate(f):
            wnid = line.strip()
            wnid_to_idx[wnid] = idx
    
    print(f"Total classes in TinyImageNet: {len(wnid_to_idx)}")
    
    # Find vehicle class indices
    vehicle_class_indices = []
    for wnid in vehicle_wnids:
        if wnid in wnid_to_idx:
            vehicle_class_indices.append(wnid_to_idx[wnid])
            print(f"Vehicle class {wnid} -> index {wnid_to_idx[wnid]}")
        else:
            print(f"Warning: Vehicle class {wnid} not found in TinyImageNet")
    
    # Count total training samples and vehicle samples
    total_samples = 0
    vehicle_samples = 0
    sample_to_class = {}  # Maps sample index to class index
    
    # Iterate through training directories
    for class_idx, wnid in enumerate(wnid_to_idx.keys()):
        class_dir = os.path.join(train_dir, wnid)
        if os.path.exists(class_dir):
            # Count images in this class
            images = [f for f in os.listdir(os.path.join(class_dir, "images")) 
                     if f.endswith(('.JPEG', '.jpg', '.png'))]
            class_sample_count = len(images)
            
            # Map sample indices to class
            for i in range(class_sample_count):
                sample_to_class[total_samples + i] = class_idx
            
            if class_idx in vehicle_class_indices:
                vehicle_samples += class_sample_count
                print(f"Vehicle class {wnid} (idx {class_idx}): {class_sample_count} samples")
            
            total_samples += class_sample_count
    
    print(f"\nTotal training samples: {total_samples}")
    print(f"Vehicle samples to forget: {vehicle_samples}")
    
    # Create binary mask
    forget_mask = torch.zeros(total_samples, dtype=torch.bool)
    
    # Mark vehicle samples for forgetting
    for sample_idx, class_idx in sample_to_class.items():
        if class_idx in vehicle_class_indices:
            forget_mask[sample_idx] = True
    
    # Save the mask
    mask_path = os.path.join(current_dir, "vehicles_forget_indices.pt")
    torch.save(forget_mask, mask_path)
    
    print(f"\n[✓] Forget mask saved to: {mask_path}")
    print(f"[✓] Mask shape: {forget_mask.shape}")
    print(f"[✓] Samples to forget: {forget_mask.sum().item()}")
    print(f"[✓] Percentage to forget: {100 * forget_mask.sum().item() / len(forget_mask):.2f}%")
    
    return mask_path

if __name__ == "__main__":
    create_vehicle_forget_mask()