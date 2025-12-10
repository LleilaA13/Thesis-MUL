#!/usr/bin/env python3
"""
Generate fresh forget masks for ResNet-50 experiments on TinyImageNet
FORMAT: Boolean tensors for compatibility with generate_mask.py and main_random.py
- DOGS: 6 classes (highly interpretable for feature visualization)
- VEHICLES: 7 classes (diverse vehicle types)
- CATS: 3 classes (for comparison with InceptionV3)
"""

import torch
import os

def get_training_samples_count():
    """Get the total number of training samples"""
    if os.path.exists('labels_tinyimagenet/train_ys.pth'):
        train_labels = torch.load('labels_tinyimagenet/train_ys.pth')
        return len(train_labels)
    else:
        # TinyImageNet standard: 200 classes × 500 samples = 100,000
        return 100000

def create_dogs_mask():
    """Create boolean forget mask for dog classes"""
    print("=== CREATING DOGS FORGET MASK (BOOLEAN) ===")
    
    # Dog classes in TinyImageNet (ImageFolder order indices)
    dog_classes = {
        11: "n02106662 - German shepherd, German shepherd dog, German police dog, alsatian",
        39: "n02099712 - Labrador retriever", 
        78: "n02099601 - golden retriever",
        135: "n02094433 - Yorkshire terrier",
        182: "n02085620 - Chihuahua",
        194: "n02113799 - standard poodle"
    }
    
    print("Dog classes to forget:")
    for idx, description in dog_classes.items():
        print(f"  Index {idx:3d}: {description}")
    
    # Create boolean mask (all False initially)
    num_samples = get_training_samples_count()
    mask = torch.zeros(num_samples, dtype=torch.bool)
    
    # Mark dog samples as True (to forget)
    samples_marked = 0
    for class_idx in dog_classes.keys():
        start_sample = class_idx * 500
        end_sample = (class_idx + 1) * 500
        mask[start_sample:end_sample] = True
        samples_marked += 500
        print(f"    Samples {start_sample:5d}-{end_sample-1:5d} marked for forgetting")
    
    # Save mask
    torch.save(mask, 'dogs_forget_mask_boolean.pt')
    
    print(f"✅ Dogs boolean mask saved: {samples_marked} samples marked for forgetting")
    print(f"   Shape: {mask.shape}, Type: {mask.dtype}")
    return samples_marked

def create_vehicles_mask():
    """Create boolean forget mask for vehicle classes"""
    print("\n=== CREATING VEHICLES FORGET MASK (BOOLEAN) ===")
    
    # Vehicle classes in TinyImageNet (ImageFolder order indices)
    vehicle_classes = {
        15: "n04146614 - school bus",
        52: "n03393912 - freight car", 
        64: "n03796401 - moving van",
        90: "n03977966 - police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria",
        117: "n04285008 - sports car, sport car",
        147: "n02814533 - beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon",
        152: "n04487081 - trolleybus, trolley coach, trackless trolley"
    }
    
    print("Vehicle classes to forget:")
    for idx, description in vehicle_classes.items():
        print(f"  Index {idx:3d}: {description}")
    
    # Create boolean mask (all False initially)
    num_samples = get_training_samples_count()
    mask = torch.zeros(num_samples, dtype=torch.bool)
    
    # Mark vehicle samples as True (to forget)
    samples_marked = 0
    for class_idx in vehicle_classes.keys():
        start_sample = class_idx * 500
        end_sample = (class_idx + 1) * 500
        mask[start_sample:end_sample] = True
        samples_marked += 500
        print(f"    Samples {start_sample:5d}-{end_sample-1:5d} marked for forgetting")
    
    # Save mask
    torch.save(mask, 'vehicles_forget_mask_boolean.pt')
    
    print(f"✅ Vehicles boolean mask saved: {samples_marked} samples marked for forgetting")
    print(f"   Shape: {mask.shape}, Type: {mask.dtype}")
    return samples_marked

def create_cats_mask():
    """Create boolean forget mask for cat classes (for ResNet-50 comparison with InceptionV3)"""
    print("\n=== CREATING CATS FORGET MASK (BOOLEAN) ===")
    
    # Cat classes in TinyImageNet (ImageFolder order indices)
    cat_classes = {
        0: "n02124075 - Egyptian cat",
        66: "n02123045 - tabby, tabby cat",
        131: "n02123394 - Persian cat"
    }
    
    print("Cat classes to forget:")
    for idx, description in cat_classes.items():
        print(f"  Index {idx:3d}: {description}")
    
    # Create boolean mask (all False initially)
    num_samples = get_training_samples_count()
    mask = torch.zeros(num_samples, dtype=torch.bool)
    
    # Mark cat samples as True (to forget)
    samples_marked = 0
    for class_idx in cat_classes.keys():
        start_sample = class_idx * 500
        end_sample = (class_idx + 1) * 500
        mask[start_sample:end_sample] = True
        samples_marked += 500
        print(f"    Samples {start_sample:5d}-{end_sample-1:5d} marked for forgetting")
    
    # Save mask
    torch.save(mask, 'cats_forget_mask_boolean.pt')
    
    print(f"✅ Cats boolean mask saved: {samples_marked} samples marked for forgetting")
    print(f"   Shape: {mask.shape}, Type: {mask.dtype}")
    print("📝 Note: This is separate from InceptionV3 cat experiments")
    return samples_marked

def verify_masks():
    """Verify all created boolean masks"""
    print("\n=== VERIFICATION ===")
    
    masks = [
        ('dogs_forget_mask_boolean.pt', 'Dogs'),
        ('vehicles_forget_mask_boolean.pt', 'Vehicles'), 
        ('cats_forget_mask_boolean.pt', 'Cats (ResNet-50)')
    ]
    
    for mask_file, name in masks:
        try:
            mask = torch.load(mask_file)
            forget_count = mask.sum().item()
            retain_count = (~mask).sum().item()
            print(f"{name}:")
            print(f"  Shape: {mask.shape}, Type: {mask.dtype}")
            print(f"  Forget samples: {forget_count}")
            print(f"  Retain samples: {retain_count}")
            print(f"  Total: {len(mask)}")
        except Exception as e:
            print(f"{name}: ERROR - {e}")

def main():
    print("🚀 GENERATING BOOLEAN FORGET MASKS FOR SALUN FRAMEWORK")
    print("=" * 70)
    
    # Check if we're in the right directory
    if not os.path.exists('datasets/tiny-imagenet-200/wnids.txt'):
        print("❌ Error: TinyImageNet dataset not found!")
        print("Please run this script from the project root directory.")
        return
    
    num_samples = get_training_samples_count()
    print(f"📊 Total training samples: {num_samples}")
    print()
    
    # Generate all boolean masks
    dogs_samples = create_dogs_mask()
    vehicles_samples = create_vehicles_mask() 
    cats_samples = create_cats_mask()
    
    # Verify
    verify_masks()
    
    print("\n" + "=" * 70)
    print("🎯 SUMMARY:")
    print(f"✅ Dogs: {dogs_samples} samples (6 classes) - BEST for feature visualization")
    print(f"✅ Vehicles: {vehicles_samples} samples (7 classes) - Good geometric features")
    print(f"✅ Cats: {cats_samples} samples (3 classes) - For comparison with InceptionV3")
    print("\n📋 READY FOR USE WITH:")
    print("  - generate_mask.py (saliency mask generation)")
    print("  - main_random.py (unlearning execution)")
    print("\n🔧 UPDATE YOUR SCRIPTS TO USE:")
    print("  --subset_indices_path dogs_forget_mask_boolean.pt")
    print("  --subset_indices_path vehicles_forget_mask_boolean.pt") 
    print("  --subset_indices_path cats_forget_mask_boolean.pt")
    print("\n🔬 InceptionV3 cat experiments (KEPT): models/inceptionv3_cat_forgetting/")

if __name__ == "__main__":
    main()