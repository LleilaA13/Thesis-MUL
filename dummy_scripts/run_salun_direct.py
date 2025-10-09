#!/usr/bin/env python3
"""
Run SalUn unlearning directly (skip mask generation)
"""

import subprocess
import os
import sys
import torch

# === GPU Configuration ===
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(f"[*] Using GPU(s): {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")

# === Setup project paths ===
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "src", "classification"))

# === Config ===
DATASET = "TinyImagenet"
ARCH = "resnet50"
MODEL_PATH = os.path.join(current_dir, "src/classification/models/resnet50_pretrained.pth")
FORGET_MASK_PATH = os.path.join(current_dir, "dogs_forget_mask_boolean.pt")
SALIENCY_DIR = os.path.join(current_dir, "masks/resnet50_dogs_forgetting")
SAVE_DIR = os.path.join(current_dir, "models/resnet50_dogs_forgetting/salun_optimal_fixed")
MASK_PATH = os.path.join(SALIENCY_DIR, "with_0.4.pt")  # Use 0.4 for optimal blocking

os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    print("🚀 RUNNING OPTIMIZED SALUN UNLEARNING")
    print("Skipping mask generation (using existing masks)")
    print()
    
    # Check prerequisites
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model not found: {MODEL_PATH}")
        return
        
    if not os.path.exists(FORGET_MASK_PATH):
        print(f"[!] Forget mask not found: {FORGET_MASK_PATH}")
        return
        
    if not os.path.exists(MASK_PATH):
        print(f"[!] Saliency mask not found: {MASK_PATH}")
        print("[*] Available masks:")
        for f in os.listdir(SALIENCY_DIR):
            if f.endswith('.pt'):
                print(f"    {f}")
        return
    
    # Load forget mask info
    try:
        forget_mask = torch.load(FORGET_MASK_PATH)
        num_to_forget = forget_mask.sum().item()
        total_samples = len(forget_mask)
        print(f"[✓] Loaded forget mask: {num_to_forget}/{total_samples} samples")
    except Exception as e:
        print(f"[!] Error loading forget mask: {e}")
        return
    
    print(f"[✓] Using saliency mask: {MASK_PATH}")
    print(f"[✓] Save directory: {SAVE_DIR}")
    
    # Create environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    
    print(f"\n[*] Running SalUn with OPTIMAL parameters:")
    print(f"    - Method: RL (Random Labels)")
    print(f"    - Learning Rate: 0.025 (5x improvement from 0.005)")
    print(f"    - Epochs: 15 (increased from 10)")
    print(f"    - Mask: 0.4 threshold (blocks ~60% of neurons)")
    print(f"    - Expected: Much better than 62% forget accuracy")
    
    try:
        subprocess.run([
            "python", os.path.join(current_dir, "src/classification/main_random.py"),
            "--unlearn", "RL",
            "--unlearn_epochs", "15",
            "--unlearn_lr", "0.025",
            "--num_indexes_to_replace", str(num_to_forget),
            "--model_path", MODEL_PATH,
            "--save_dir", SAVE_DIR,
            "--mask_path", MASK_PATH,
            "--subset_indices_path", FORGET_MASK_PATH,
            "--arch", ARCH,
            "--dataset", DATASET,
            "--imagenet_arch",
            "--train_y_file", os.path.join(current_dir, "labels_tinyimagenet", "train_ys.pth"),
            "--val_y_file", os.path.join(current_dir, "labels_tinyimagenet", "val_ys.pth")
        ], check=True, env=env)
        
        print(f"\n[✅] SalUn unlearning completed successfully!")
        print(f"[*] Model saved to: {SAVE_DIR}")
        print(f"\n[*] Next steps:")
        print(f"1. Test results: python test_unlearned_model.py")
        print(f"2. View logs: python results_tracker.py view dogs")
        
    except subprocess.CalledProcessError as e:
        print(f"[❌] Unlearning failed: {e}")
        print(f"[*] Check if the environment has all required packages")
    except Exception as e:
        print(f"[❌] Unexpected error: {e}")

if __name__ == "__main__":
    main()