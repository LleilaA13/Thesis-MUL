#!/usr/bin/env python3
"""
Generate forget masks in the correct format for generate_mask.py and main_random.py
Expected format: Boolean tensor [True, False, True, ...] of length = num_training_samples
"""

import torch
import os

def create_boolean_masks():
    """Create boolean masks in the format expected by the SalUn framework"""
    
    print("🔧 CREATING BOOLEAN MASKS FOR SALUN FRAMEWORK")
    print("=" * 60)
    
    # Load TinyImageNet labels to get the number of training samples
    train_labels = torch.load('labels_tinyimagenet/train_ys.pth')
    num_samples = len(train_labels)  # Should be 100,000 for TinyImageNet
    
    print(f"📊 Total training samples: {num_samples}")
    print()
    
    # Define our class groups
    class_groups = {
        'dogs': {
            'indices': [11, 39, 78, 135, 182, 194],
            'descriptions': [
                "German shepherd", "Labrador retriever", "golden retriever",
                "Yorkshire terrier", "Chihuahua", "standard poodle"
            ]
        },
        'vehicles': {
            'indices': [15, 52, 64, 90, 117, 147, 152], 
            'descriptions': [
                "school bus", "freight car", "moving van", "police van",
                "sports car", "station wagon", "trolleybus"
            ]
        },
        'cats': {
            'indices': [0, 66, 131],
            'descriptions': [
                "Egyptian cat", "tabby cat", "Persian cat"
            ]
        }
    }
    
    # Generate boolean masks for each group
    for group_name, group_info in class_groups.items():
        print(f"🎯 Creating {group_name.upper()} mask:")
        
        # Create boolean mask (all False initially)
        mask = torch.zeros(num_samples, dtype=torch.bool)
        
        # Mark samples from target classes as True (to forget)
        samples_marked = 0
        for class_idx in group_info['indices']:
            # Each class has 500 samples, starting at class_idx * 500
            start_idx = class_idx * 500
            end_idx = start_idx + 500
            mask[start_idx:end_idx] = True
            samples_marked += 500
            print(f"   Class {class_idx:3d}: samples {start_idx:5d}-{end_idx-1:5d} marked for forgetting")
        
        # Save the mask
        mask_file = f"{group_name}_forget_mask_boolean.pt"
        torch.save(mask, mask_file)
        
        # Verify
        print(f"   ✅ Saved: {mask_file}")
        print(f"   📊 Total samples marked: {mask.sum().item()}")
        print(f"   📊 Total samples retained: {(~mask).sum().item()}")
        print()
    
    print("🔍 VERIFICATION:")
    print("=" * 60)
    
    # Verify all masks
    for group_name in class_groups.keys():
        mask_file = f"{group_name}_forget_mask_boolean.pt"
        mask = torch.load(mask_file)
        
        print(f"{group_name.upper()} mask:")
        print(f"  Shape: {mask.shape}")
        print(f"  Type: {mask.dtype}")
        print(f"  Forget samples: {mask.sum().item()}")
        print(f"  Retain samples: {(~mask).sum().item()}")
        print()
    
    print("✅ ALL MASKS CREATED IN CORRECT FORMAT!")
    print("Ready for use with generate_mask.py and main_random.py")
    
    return True

if __name__ == "__main__":
    if not os.path.exists('labels_tinyimagenet/train_ys.pth'):
        print("❌ Error: TinyImageNet labels not found!")
        print("Please run create_tinyimagenet_labels.py first.")
    else:
        create_boolean_masks()