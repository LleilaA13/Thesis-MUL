#!/usr/bin/env python3
"""
Debug SalUn implementation based on official repository findings
"""

import torch
import numpy as np
import sys
import os
sys.path.append('src/classification')

def debug_salun_implementation():
    """Debug common SalUn implementation issues"""
    
    print("=== SALUN IMPLEMENTATION DEBUGGING ===\n")
    
    print("🔍 CHECKING COMMON ISSUES:")
    
    # 1. Check if masks exist and are being loaded
    mask_path = "masks/resnet50_dogs_forgetting/mask0_5.pt"  # Adjust path
    if os.path.exists(mask_path):
        print(f"✅ Mask file found: {mask_path}")
        mask = torch.load(mask_path, map_location='cpu')
        print(f"   Mask contains {len(mask)} layers")
        
        # Check mask statistics
        total_params = 0
        blocked_params = 0
        for layer_name, layer_mask in mask.items():
            total_params += layer_mask.numel()
            blocked_params += (layer_mask == 0).sum().item()
        
        block_ratio = blocked_params / total_params
        print(f"   Blocking {block_ratio:.1%} of parameters ({blocked_params:,}/{total_params:,})")
        
        if block_ratio < 0.3:
            print(f"⚠️  LOW BLOCKING RATIO: Only {block_ratio:.1%} blocked (should be 30-70%)")
        elif block_ratio > 0.8:
            print(f"⚠️  HIGH BLOCKING RATIO: {block_ratio:.1%} blocked (may damage model too much)")
        else:
            print(f"✅ GOOD BLOCKING RATIO: {block_ratio:.1%} blocked")
            
    else:
        print(f"❌ Mask file not found: {mask_path}")
        print("   → Generate masks first: python generate_mask.py")
    
    print()
    
    # 2. Check unlearning configuration 
    print("🎯 SALUN HYPERPARAMETER RECOMMENDATIONS:")
    print("Based on official SalUn repository analysis:")
    print()
    
    print("✅ SUCCESSFUL PARAMETERS:")
    print(f"   Learning Rate: 0.01 - 0.05 (you used 0.005 in failed run)")
    print(f"   Epochs: 10-15")
    print(f"   Mask Threshold: 0.3-0.5 (0.3 = block 70% neurons)")
    print(f"   Method: RL (Random Labels)")
    print(f"   Batch Size: 256")
    print()
    
    print("❌ YOUR RECENT FAILED RUN:")
    print(f"   Learning Rate: 0.005 ← TOO LOW!")
    print(f"   Epochs: 10 ← OK")  
    print(f"   Mask Threshold: 0.5 ← OK")
    print(f"   Result: 62% forget acc ← FAILED")
    print()
    
    print("🎯 RECOMMENDED NEXT RUN:")
    print(f"   Learning Rate: 0.02-0.03")
    print(f"   Epochs: 12-15") 
    print(f"   Mask Threshold: 0.4")
    print(f"   Expected: <50% forget acc")
    
    print()
    
    # 3. Check evaluation logic
    print("🔍 EVALUATION DEBUGGING:")
    print("Common issues with forget accuracy evaluation:")
    print()
    print("1. ❌ LABEL RESTORATION BUG:")
    print("   → Make sure forget samples use ORIGINAL dog labels during evaluation")
    print("   → NOT the random labels used during training")
    print()
    print("2. ❌ WRONG FORGET SET:")
    print("   → Ensure evaluation uses same forget samples as training")
    print("   → Check dogs_forget_indices.pt matches actual forget set")
    print()
    print("3. ❌ MASK NOT REMOVED:")
    print("   → Remove saliency masks during evaluation (they're only for training)")
    print()
    
    # 4. Official SalUn results for comparison
    print("📊 OFFICIAL SALUN RESULTS (for comparison):")
    print("From the paper, SalUn achieved:")
    print("   CIFAR-10 class unlearning: ~98% unlearn accuracy (2% forget acc)")
    print("   ImageNet class unlearning: ~95% unlearn accuracy (5% forget acc)")
    print()
    print("Your target: <50% forget acc (preferably <30%)")
    
def suggest_next_experiments():
    """Suggest specific experiments based on SalUn findings"""
    
    print("\n🚀 SPECIFIC NEXT EXPERIMENTS:")
    print()
    
    print("1. 🔥 FOLLOW OFFICIAL SALUN PARAMS:")
    print("   Based on successful runs in their repo:")
    cmd1 = "python resnet50_unlearn_dogs.py --epochs 15 --lr 0.03 --mask_threshold 0.4"
    print(f"   {cmd1}")
    print()
    
    print("2. 🧪 TRY GRADIENT ASCENT (GA) METHOD:")
    print("   GA maximizes loss on forget data (may work better than RL):")
    cmd2 = "python resnet50_unlearn_dogs.py --method GA --epochs 12 --lr 0.02"
    print(f"   {cmd2}")
    print()
    
    print("3. 🎯 VERIFY MASK QUALITY:")
    print("   Generate new masks with different thresholds:")
    cmd3 = "python generate_mask.py --threshold 0.3  # Block 70% neurons"
    print(f"   {cmd3}")
    print()
    
    print("4. 🔍 DEBUG EVALUATION:")
    print("   Check if evaluation logic matches SalUn paper:")
    print("   → Forget samples should use ORIGINAL labels")
    print("   → Remove masks during evaluation")
    print("   → Use same data split as training")

def check_implementation_details():
    """Check specific implementation details"""
    
    print("\n🔧 IMPLEMENTATION CHECKLIST:")
    print()
    
    print("✅ Check these in your code:")
    print()
    print("1. RANDOM LABEL ASSIGNMENT (RL method):")
    print("   for batch in forget_loader:")
    print("       target = torch.randint(0, num_classes, target.shape)")
    print("       # Train with random targets, not original dog labels")
    print()
    
    print("2. MASK APPLICATION:")
    print("   loss.backward()")
    print("   if mask:")
    print("       for name, param in model.named_parameters():")
    print("           if param.grad is not None:")
    print("               param.grad *= mask[name]  # Zero out gradients")
    print("   optimizer.step()")
    print()
    
    print("3. EVALUATION (NO MASKS):")
    print("   model.eval()")
    print("   # Don't apply masks during evaluation!")
    print("   with torch.no_grad():")
    print("       output = model(forget_data)")
    print("       # Use ORIGINAL dog labels for accuracy")
    print()
    
    print("4. HYPERPARAMETERS:")
    print("   lr = 0.02-0.03  # Higher than 0.005")
    print("   epochs = 12-15  # Sufficient time")
    print("   mask_threshold = 0.3-0.4  # Block 60-70%")

if __name__ == "__main__":
    debug_salun_implementation()
    suggest_next_experiments() 
    check_implementation_details()