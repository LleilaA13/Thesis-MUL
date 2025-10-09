#!/usr/bin/env python3
"""
Quick debugging script to understand the Random Labels method and forget accuracy
"""
import torch
import numpy as np
import sys
import os

# Add the path to access the dataset classes
sys.path.append('src/classification')

def analyze_forget_accuracy_concept():
    """
    Explain what forget accuracy means in the context of Random Labels unlearning
    """
    print("=== Understanding Forget Accuracy in Random Labels Unlearning ===\n")
    
    print("1. TRAINING PHASE:")
    print("   - Original labels for dog samples: [dog_class_id, dog_class_id, ...]")
    print("   - Random labels for dog samples:   [random_1, random_2, ...]")
    print("   - Model learns: dog_image -> random_label (incorrect association)")
    print()
    
    print("2. EVALUATION PHASE:")
    print("   - Restore original labels for dog samples")
    print("   - Test model on: dog_image -> dog_class_id (correct association)")
    print("   - If unlearning worked: model should perform poorly (low accuracy)")
    print("   - If unlearning failed: model still remembers (high accuracy)")
    print()
    
    print("3. EXPECTED RESULTS:")
    print("   - Forget accuracy: ~10-30% (random chance for dog classes)")
    print("   - Retain accuracy: ~80-90% (model should still work on other classes)")
    print("   - If forget accuracy is 70%+: UNLEARNING FAILED")
    print()

def check_mask_distribution():
    """Check the distribution of samples in our forget mask"""
    print("=== Forget Mask Analysis ===\n")
    
    # Load the mask
    try:
        mask = torch.load('dogs_forget_mask_boolean.pt')
        print(f"Mask shape: {mask.shape}")
        print(f"Total samples: {len(mask)}")
        print(f"Forget samples: {mask.sum().item()}")
        print(f"Retain samples: {(~mask).sum().item()}")
        print(f"Forget percentage: {100 * mask.sum().item() / len(mask):.1f}%")
        
        # Check if mask is reasonable (should be around 3% for dogs)
        expected_dog_samples = 6 * 500  # 6 dog classes * 500 samples each
        actual_forget_samples = mask.sum().item()
        
        print(f"\nExpected dog samples: {expected_dog_samples}")
        print(f"Actual forget samples: {actual_forget_samples}")
        
        if actual_forget_samples == expected_dog_samples:
            print("✅ Mask looks correct!")
        else:
            print("❌ Mask might be incorrect!")
            
    except Exception as e:
        print(f"Error loading mask: {e}")

def simulate_random_accuracy():
    """Simulate what random accuracy should be for dog classes"""
    print("\n=== Random Accuracy Simulation ===\n")
    
    # TinyImageNet has 200 classes
    num_classes = 200
    num_dog_classes = 6
    
    # If model outputs random predictions
    random_accuracy = 1.0 / num_classes * 100
    
    # If model is biased toward dog classes
    dog_class_bias_accuracy = num_dog_classes / num_classes * 100
    
    print(f"Pure random accuracy: {random_accuracy:.1f}%")
    print(f"Dog-class-biased accuracy: {dog_class_bias_accuracy:.1f}%")
    print(f"Current observed: ~70% (UNLEARNING FAILED)")
    print()
    print("Conclusion: 70% >> 3% indicates model still strongly remembers dog features")

if __name__ == "__main__":
    analyze_forget_accuracy_concept()
    check_mask_distribution()
    simulate_random_accuracy()