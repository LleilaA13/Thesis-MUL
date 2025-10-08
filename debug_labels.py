#!/usr/bin/env python3
"""
Debug script to check what labels are being used during unlearning
"""
import torch
import sys
import os
sys.path.append('src/classification')

def debug_labels():
    """Check the labels in various dataset components"""
    
    # Load the trained models and datasets to see what happened
    model_path = "/media/hdd/usr/leyla/Unlearn-Saliency/models/resnet50_dogs_forgetting/mask0_5/RLcheckpoint.pth.tar"
    eval_path = "/media/hdd/usr/leyla/Unlearn-Saliency/models/resnet50_dogs_forgetting/mask0_5/RLeval_result.pth.tar"
    
    if os.path.exists(eval_path):
        eval_results = torch.load(eval_path, map_location='cpu')
        print("=== Evaluation Results ===")
        print(f"Accuracy: {eval_results.get('accuracy', 'N/A')}")
        
        # Check SVC MIA results
        if 'SVC_MIA_forget_efficacy' in eval_results:
            mia = eval_results['SVC_MIA_forget_efficacy']
            print(f"MIA Results: {mia}")
    
    # Load forget mask to see which samples should be forgotten
    forget_mask = torch.load('/media/hdd/usr/leyla/Unlearn-Saliency/dogs_forget_mask_boolean.pt')
    forget_indices = torch.where(forget_mask)[0]
    print(f"\n=== Forget Mask Info ===")
    print(f"Total samples: {len(forget_mask)}")
    print(f"Forget samples: {forget_mask.sum().item()}")
    print(f"First 10 forget indices: {forget_indices[:10].tolist()}")
    
    # The key insight: 
    print(f"\n=== Analysis ===")
    print(f"High forget accuracy (70.67%) suggests:")
    print(f"1. Model still recognizes dog patterns")
    print(f"2. Random labels may not be applied correctly")
    print(f"3. Or evaluation uses wrong dataset")
    
    # Check which classes are dogs (first 6 classes in TinyImageNet ordering)
    print(f"\n=== Dog Classes ===")
    print(f"Dog classes should be: 0, 1, 2, 3, 4, 5 (first 6 classes)")
    print(f"Forget samples: {forget_mask.sum().item()} should be mostly from these classes")
    
    # Calculate expected random accuracy
    print(f"\n=== Expected Results ===")
    print(f"Random chance accuracy for 200 classes: {100/200:.1f}%")
    print(f"Current forget accuracy: 70.67% (way too high!)")
    print(f"This suggests unlearning failed - model still knows dogs")

if __name__ == "__main__":
    debug_labels()