#!/usr/bin/env python3
"""
Comprehensive status report for SalUn unlearning experiments
"""

import os
import torch
from unlearn_config import TINYIMAGENET_CLASSES

def print_header(title):
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)

def check_config_status():
    print_header("CONFIGURATION STATUS")
    
    print("✅ AVAILABLE FORGET CATEGORIES:")
    for forget_type, config in TINYIMAGENET_CLASSES.items():
        print(f"   {forget_type.upper()}: {len(config['wnids'])} classes")
        print(f"     - Sample count: {len(config['indices']) * 500} training samples")
        print(f"     - Classes: {config['names']}")
        print()

def check_mask_status():
    print_header("FORGET MASK STATUS")
    
    masks = {
        'dogs': 'dogs_forget_mask_boolean.pt',
        'cats': 'cats_forget_mask_boolean.pt', 
        'vehicles': 'vehicles_forget_mask_boolean.pt'
    }
    
    for forget_type, mask_file in masks.items():
        if os.path.exists(mask_file):
            mask = torch.load(mask_file)
            expected = len(TINYIMAGENET_CLASSES[forget_type]['indices']) * 500
            actual = mask.sum().item()
            status = "✅" if actual == expected else "❌"
            print(f"{status} {forget_type.upper()}: {mask_file}")
            print(f"     Expected: {expected}, Actual: {actual}, Total: {len(mask)}")
        else:
            print(f"❌ {forget_type.upper()}: {mask_file} - NOT FOUND")
        print()

def check_saliency_mask_status():
    print_header("SALIENCY MASK STATUS")
    
    saliency_dirs = {
        'dogs': 'masks/resnet50_dogs_forgetting',
        'cats': 'masks/resnet50_cats_forgetting',
        'vehicles': 'masks/resnet50_vehicles_forgetting'
    }
    
    for forget_type, mask_dir in saliency_dirs.items():
        mask_file = os.path.join(mask_dir, 'with_0.5.pt')
        if os.path.exists(mask_file):
            print(f"✅ {forget_type.upper()}: {mask_file}")
            # Quick stats
            mask = torch.load(mask_file)
            total_params = sum(v.numel() for v in mask.values())
            masked_params = sum((v == 0).sum().item() for v in mask.values())
            print(f"     {masked_params}/{total_params} ({100*masked_params/total_params:.1f}%) parameters masked")
        else:
            print(f"❌ {forget_type.upper()}: {mask_file} - NOT FOUND")
        print()

def check_script_status():
    print_header("SCRIPT STATUS")
    
    scripts = {
        'dogs': 'resnet50_unlearn_dogs.py',
        'cats': 'resnet50_unlearn_cats.py',
        'vehicles': 'resnet50_unlearn_vehicles.py'
    }
    
    for forget_type, script_file in scripts.items():
        if os.path.exists(script_file):
            print(f"✅ {forget_type.upper()}: {script_file}")
            
            # Check for key configuration
            with open(script_file, 'r') as f:
                content = f.read()
                if '--unlearn_epochs", "5"' in content:
                    epochs_status = "✅ 5 epochs"
                else:
                    epochs_status = "❌ Wrong epochs"
                    
                if '--unlearn_lr", "0.01"' in content:
                    lr_status = "✅ 0.01 LR"
                else:
                    lr_status = "❌ Wrong LR"
                    
                print(f"     {epochs_status}, {lr_status}")
        else:
            print(f"❌ {forget_type.upper()}: {script_file} - NOT FOUND")
        print()

def print_fixes_summary():
    print_header("FIXES IMPLEMENTED")
    
    fixes = [
        "✅ Created centralized unlearn_config.py with correct class mappings",
        "✅ Fixed dog classes: 6 classes (Chihuahua, Yorkshire Terrier, etc.)",
        "✅ Added cat classes: 4 classes (Egyptian Cat, Tabby Cat, etc.)",  
        "✅ Added vehicle classes: 8 classes (Car, Bus, Taxi, etc.)",
        "✅ Fixed RL.py random label assignment for Subset datasets",
        "✅ Added original label storage for proper evaluation",
        "✅ Fixed transform duplication issues (ToTensor)",
        "✅ Updated main_random.py to auto-detect forget type",
        "✅ Increased unlearn_epochs from 3 to 5",
        "✅ Increased unlearn_lr from 0.001 to 0.01",
        "✅ Fixed label file paths to use labels_tinyimagenet",
        "✅ Created boolean forget masks with correct sample counts"
    ]
    
    for fix in fixes:
        print(f"  {fix}")
    print()

def print_expected_results():
    print_header("EXPECTED RESULTS AFTER FIXES")
    
    print("📊 FORGET ACCURACY EXPECTATIONS:")
    print("   - BEFORE: ~70% (unlearning failed)")  
    print("   - AFTER:  ~0.5-5% (successful unlearning)")
    print("   - Random chance: 0.5% (1/200 classes)")
    print()
    
    print("📊 RETAIN ACCURACY EXPECTATIONS:")
    print("   - Should remain: ~80-90% (model still works on other classes)")
    print()
    
    print("🔍 KEY INDICATORS OF SUCCESS:")
    print("   1. Forget accuracy drops dramatically (70% → <5%)")
    print("   2. Retain accuracy stays high (>80%)")
    print("   3. Training shows random label assignment working")
    print("   4. Evaluation shows label restoration working")
    print()

def print_usage_instructions():
    print_header("USAGE INSTRUCTIONS")
    
    print("🚀 TO RUN EXPERIMENTS:")
    print("   conda activate salUN")
    print("   python resnet50_unlearn_dogs.py      # Test dogs")
    print("   python resnet50_unlearn_cats.py      # Test cats") 
    print("   python resnet50_unlearn_vehicles.py  # Test vehicles")
    print()
    
    print("📝 TO MONITOR PROGRESS:")
    print("   - Look for '[DEBUG] Assigned random labels to X forget samples'")
    print("   - Training accuracy should be reasonable (not 100%)")
    print("   - Final forget accuracy should be <5%")
    print()

if __name__ == "__main__":
    check_config_status()
    check_mask_status()
    check_saliency_mask_status()
    check_script_status()
    print_fixes_summary()
    print_expected_results()
    print_usage_instructions()