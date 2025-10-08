#!/usr/bin/env python3
"""
Quick test to evaluate the unlearned model from the recent run
"""
import torch
import torch.nn as nn
from torchvision import transforms
import sys
import os
sys.path.append('src/classification')

# Add the classification directory to path
from dataset import TinyImageNet
import models

def test_unlearned_model():
    """Test the unlearned model on dog classes"""
    
    # Load the unlearned model
    model_path = "/media/hdd/usr/leyla/Unlearn-Saliency/models/resnet50_dogs_forgetting/mask0_5/RLcheckpoint.pth.tar"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
        
    print(f"[*] Loading unlearned model from {model_path}")
    
    # Create model architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(num_classes=200, imagenet=True)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✅ Model loaded successfully!")
    
    # Simple analysis: Check if the model has actually changed
    # Load the original pretrained model for comparison
    orig_model_path = "/media/hdd/usr/leyla/Unlearn-Saliency/src/classification/models/resnet50_pretrained.pth"
    if os.path.exists(orig_model_path):
        print(f"[*] Loading original model for comparison...")
        orig_model = models.resnet50(num_classes=200, imagenet=True)
        orig_model = orig_model.to(device)
        
        orig_checkpoint = torch.load(orig_model_path, map_location=device)
        if 'state_dict' in orig_checkpoint:
            orig_model.load_state_dict(orig_checkpoint['state_dict'])
        else:
            orig_model.load_state_dict(orig_checkpoint)
        
        orig_model.eval()
        
        # Compare some key parameters
        unlearned_fc_weight = model.fc.weight.data
        original_fc_weight = orig_model.fc.weight.data
        
        weight_diff = torch.norm(unlearned_fc_weight - original_fc_weight)
        print(f"[*] Final layer weight difference: {weight_diff:.6f}")
        
        if weight_diff > 0.01:
            print("✅ Model weights have changed significantly - unlearning process ran")
        else:
            print("❌ Model weights barely changed - unlearning may have failed")
    
    # Load evaluation results if available
    eval_path = "/media/hdd/usr/leyla/Unlearn-Saliency/models/resnet50_dogs_forgetting/mask0_5/RLeval_result.pth.tar"
    if os.path.exists(eval_path):
        print(f"[*] Loading evaluation results...")
        eval_results = torch.load(eval_path, map_location='cpu')
        print(f"[*] Evaluation results: {eval_results}")
        
        if 'accuracy' in eval_results:
            for key, acc in eval_results['accuracy'].items():
                print(f"    {key} accuracy: {acc:.2f}%")
    
    print(f"\n[*] Based on your terminal output:")
    print(f"    forget_test acc: 70.67% ❌ (should be ~0.5% for successful unlearning)")
    print(f"    retain_test acc: 67.23% ✅ (reasonable performance on non-dog classes)")
    
    print(f"\n[*] Analysis:")
    print(f"    The high forget accuracy (70.67%) indicates that the model")
    print(f"    still remembers dog classes well, suggesting unlearning failed.")
    print(f"    This could be due to:")
    print(f"    1. Evaluation logic restoring forget labels incorrectly")
    print(f"    2. RL method not working properly with SalUn saliency masks")
    print(f"    3. Insufficient unlearning epochs or wrong hyperparameters")

if __name__ == "__main__":
    test_unlearned_model()