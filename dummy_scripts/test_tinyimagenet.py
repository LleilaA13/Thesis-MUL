#!/usr/bin/env python3
"""
Minimal test of TinyImageNet dataset loading
"""

import os
import sys
import torch
from torchvision import transforms

# Add the src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the project 'src' root so we can import classification.dataset
sys.path.append(os.path.join(current_dir, "src"))

try:
    from classification.dataset import TinyImageNet
except ImportError as e:
    raise ImportError("Could not import TinyImageNet from classification.dataset. Ensure src/classification/dataset.py exists.") from e

class Args:
    def __init__(self):
        self.data_dir = "/media/hdd/usr/leyla/Unlearn-Saliency/datasets/tiny-imagenet-200"
        self.batch_size = 32

def test_tinyimagenet():
    print("Testing TinyImageNet dataset loading...")
    
    args = Args()
    dataset = TinyImageNet(args)
    
    try:
        train_loader, val_loader, test_loader = dataset.data_loaders(
            batch_size=32,
            num_workers=0  # Use 0 workers for testing
        )
        
        print(f"✓ Success! Train loader created with {len(train_loader.dataset)} samples")
        print(f"✓ Val loader created with {len(val_loader.dataset)} samples") 
        print(f"✓ Test loader created with {len(test_loader.dataset)} samples")
        
        # Test loading one batch
        print("Testing batch loading...")
        batch_iter = iter(train_loader)
        images, labels = next(batch_iter)
        print(f"✓ Successfully loaded batch: {images.shape}, labels: {labels.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tinyimagenet()
    sys.exit(0 if success else 1)