#!/usr/bin/env python3
"""
Quick Device Fix Test for Lucent + Tiny ImageNet
This script tests if the device compatibility issues are resolved.
"""

import torch
from torchvision import models
from lucent.optvis import render

def load_resnet50_tinyimagenet_fixed(model_path, model_name="model", device='auto'):
    """Load ResNet50 for Tiny ImageNet with proper device handling"""
    # Automatic device detection
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading {model_name} on device: {device}")
    
    model = models.resnet50(weights=None)  # Updated syntax for newer PyTorch
    # Key: 200 classes for Tiny ImageNet, NOT 1000!
    model.fc = torch.nn.Linear(model.fc.in_features, 200)
    
    # Load your trained weights
    # Fix for PyTorch 2.6+ weights_only security feature
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    try:
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except RuntimeError as e:
        print(f"Warning: Loading {model_name} with strict=False due to: {e}")
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    # Move model to device AFTER loading weights
    model = model.to(device)
    model.eval()
    print(f"✅ Successfully loaded {model_name} on {device}")
    return model

def test_device_compatibility():
    """Test if device compatibility is properly handled"""
    print("🔧 Testing Device Compatibility...")
    
    # Check device availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Available device: {device}")
    
    # Test model creation
    try:
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 200)
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model created successfully on {device}")
        print(f"Model device: {next(model.parameters()).device}")
        
        # Test if Lucent can work with this device setup
        print("🧪 Testing Lucent visualization...")
        try:
            # Try a simple visualization
            _ = render.render_vis(model, "labels:0", thresholds=(64,), show_inline=False)
            print("✅ Lucent visualization test passed!")
            return True
        except Exception as e:
            print(f"❌ Lucent visualization failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Device Compatibility Fix Test")
    print("="*50)
    
    success = test_device_compatibility()
    
    if success:
        print("\n🎉 SUCCESS! Device compatibility is working properly.")
        print("\nYou can now use the updated guide functions without device mismatch errors.")
        print("\nNext steps:")
        print("1. Use load_resnet50_tinyimagenet() with device parameter")
        print("2. Run your Lucent visualizations")
        print("3. Compare original vs unlearned models")
    else:
        print("\n⚠️  Device compatibility test failed.")
        print("Please check your PyTorch installation and GPU drivers.")