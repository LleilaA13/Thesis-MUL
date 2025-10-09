#!/usr/bin/env python3
"""
Advanced SalUn unlearning script with optimized parameters
Based on official SalUn repository analysis
"""

import subprocess
import os
import sys
import torch
from tqdm import tqdm

# === GPU Configuration ===
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(f"[*] Using GPU(s): {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")

# === Setup project paths ===
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "src", "classification"))

# === Advanced SalUn Configuration ===
DATASET = "TinyImagenet"
ARCH = "resnet50"
TIN_IMAGENET_DIR = os.path.join(current_dir, "datasets/tiny-imagenet-200")
MODEL_PATH = os.path.join(current_dir, "src/classification/models/resnet50_pretrained.pth")
FORGET_MASK_PATH = os.path.join(current_dir, "dogs_forget_mask_boolean.pt")

# Multiple experiment configurations based on SalUn analysis
EXPERIMENTS = [
    {
        "name": "salun_optimal", 
        "lr": 0.025, 
        "epochs": 15, 
        "mask_threshold": 0.4,
        "method": "RL",
        "description": "Optimal params from SalUn analysis"
    },
    {
        "name": "salun_aggressive", 
        "lr": 0.04, 
        "epochs": 18, 
        "mask_threshold": 0.3,
        "method": "RL", 
        "description": "More aggressive - block 70% neurons"
    },
    {
        "name": "salun_ga_method", 
        "lr": 0.02, 
        "epochs": 12, 
        "mask_threshold": 0.4,
        "method": "GA",
        "description": "Gradient Ascent instead of Random Labels"
    }
]

def run_experiment(exp_config):
    """Run a single unlearning experiment"""
    
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING EXPERIMENT: {exp_config['name'].upper()}")
    print(f"📝 Description: {exp_config['description']}")
    print(f"⚙️  Parameters: LR={exp_config['lr']}, epochs={exp_config['epochs']}, mask={exp_config['mask_threshold']}, method={exp_config['method']}")
    print(f"{'='*60}")
    
    # Setup directories
    SALIENCY_DIR = os.path.join(current_dir, f"masks/resnet50_dogs_forgetting_{exp_config['name']}")
    SAVE_DIR = os.path.join(current_dir, f"models/resnet50_dogs_forgetting/{exp_config['name']}")
    MASK_PATH = os.path.join(SALIENCY_DIR, f"with_{exp_config['mask_threshold']}.pt")
    
    os.makedirs(SALIENCY_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Load forget mask to get sample count
    try:
        forget_mask = torch.load(FORGET_MASK_PATH)
        num_to_forget = forget_mask.sum().item()
        total_samples = len(forget_mask)
        print(f"[✓] Loaded forget mask: {num_to_forget}/{total_samples} samples")
    except Exception as e:
        print(f"[!] Error loading forget mask: {e}")
        return False
    
    # Create environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    
    try:
        # Step 1: Generate saliency mask with custom threshold
        print(f"[*] Generating saliency mask (threshold={exp_config['mask_threshold']})...")
        
        subprocess.run([
            "python", os.path.join(current_dir, "src/classification/generate_mask.py"),
            "--save_dir", SALIENCY_DIR,
            "--model_path", MODEL_PATH,
            "--subset_indices_path", FORGET_MASK_PATH,
            "--num_indexes_to_replace", str(num_to_forget),
            "--unlearn_epochs", "1",
            "--arch", ARCH,
            "--dataset", DATASET,
            "--imagenet_arch",
            "--train_y_file", os.path.join(current_dir, "labels_tinyimagenet", "train_ys.pth"),
            "--val_y_file", os.path.join(current_dir, "labels_tinyimagenet", "val_ys.pth")
        ], check=True, env=env)
        
        # Update mask path based on actual generated file
        # The generate_mask.py might create different threshold files
        possible_masks = [
            os.path.join(SALIENCY_DIR, f"with_{exp_config['mask_threshold']}.pt"),
            os.path.join(SALIENCY_DIR, "with_0.5.pt"),  # Default fallback
        ]
        
        MASK_PATH = None
        for mask_path in possible_masks:
            if os.path.exists(mask_path):
                MASK_PATH = mask_path
                break
                
        if MASK_PATH is None:
            print(f"[!] No mask file found in {SALIENCY_DIR}")
            return False
            
        print(f"[✓] Using mask: {MASK_PATH}")
        
        # Step 2: Run unlearning with optimized parameters
        print(f"[*] Running {exp_config['method']} unlearning...")
        
        subprocess.run([
            "python", os.path.join(current_dir, "src/classification/main_random.py"),
            "--unlearn", exp_config['method'],
            "--unlearn_epochs", str(exp_config['epochs']),
            "--unlearn_lr", str(exp_config['lr']),
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
        
        print(f"[✅] Experiment {exp_config['name']} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[❌] Experiment {exp_config['name']} failed: {e}")
        return False
    except Exception as e:
        print(f"[❌] Unexpected error in {exp_config['name']}: {e}")
        return False

def main():
    """Run multiple SalUn experiments with different configurations"""
    
    print("🧪 ADVANCED SALUN UNLEARNING EXPERIMENTS")
    print("Based on official SalUn repository analysis\n")
    
    # Check prerequisites
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model not found: {MODEL_PATH}")
        print("[*] Please ensure you have a trained ResNet-50 model")
        return
        
    if not os.path.exists(FORGET_MASK_PATH):
        print(f"[!] Forget mask not found: {FORGET_MASK_PATH}")
        print("[*] Please create the dog forget mask first")
        return
    
    # Ask user which experiments to run
    print("Available experiments:")
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"  {i}. {exp['name']}: {exp['description']}")
    
    choice = input(f"\nSelect experiment (1-{len(EXPERIMENTS)}) or 'all' for all experiments: ").strip().lower()
    
    if choice == 'all':
        experiments_to_run = EXPERIMENTS
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(EXPERIMENTS):
                experiments_to_run = [EXPERIMENTS[idx]]
            else:
                print("Invalid choice")
                return
        except ValueError:
            print("Invalid choice")
            return
    
    # Run selected experiments
    results = {}
    for exp_config in experiments_to_run:
        success = run_experiment(exp_config)
        results[exp_config['name']] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    for exp_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{exp_name}: {status}")
    
    print(f"\n[*] Check results in models/resnet50_dogs_forgetting/ directories")
    print(f"[*] Use test_unlearned_model.py to evaluate the results")
    print(f"[*] Log results using: python results_tracker.py log")

if __name__ == "__main__":
    main()