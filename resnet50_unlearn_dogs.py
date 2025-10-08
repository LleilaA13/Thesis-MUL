import subprocess
import os
import sys
import torch
from tqdm import tqdm

# === GPU Configuration ===
# Set which GPU to use 

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1 which has more free memory
print(f"[*] Using GPU(s): {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")

# === Setup project paths ===
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "src", "classification"))

# === Config ===
# Example: Forgetting dog classes (replace with actual class indices)
dogs_wnids = [
    "n02106662", "n02099712", "n02099601", "n02094433", "n02085620",  "n02113799"
]  # Fill in with correct indices
DATASET = "TinyImagenet"
ARCH = "resnet50"
TIN_IMAGENET_DIR = os.path.join(current_dir, "datasets/tiny-imagenet-200")
MODEL_PATH = os.path.join(current_dir, "src/classification/models/resnet50_pretrained.pth")  # Update to correct path
FORGET_MASK_PATH = os.path.join(current_dir, "dogs_forget_mask_boolean.pt")
SALIENCY_DIR = os.path.join(current_dir, "masks/resnet50_dogs_forgetting")
SAVE_DIR = os.path.join(current_dir, "models/resnet50_dogs_forgetting/mask0_5_salun")
RESULTS_DIR = os.path.join(current_dir, "results/resnet50_dogs_forgetting")
MASK_PATH = os.path.join(SALIENCY_DIR, "with_0.5.pt")  # SalUn uses 0.5 as standard

os.makedirs(SALIENCY_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Step 1: Check or create the forget mask ===
# First, check if the forget mask exists, if not, create it
if not os.path.exists(FORGET_MASK_PATH):
    print(f"\n[!] Forget mask not found at {FORGET_MASK_PATH}")
    print("[*] Creating dog forget mask...")
    
    # Import and run the helper function
    helper_script = os.path.join(current_dir, "create_dog_forget_mask.py")
    subprocess.run(["python", helper_script], check=True)

# === Step 2: Train or load base model ===
if not os.path.exists(MODEL_PATH):
    print(f"\n[!] Model not found at {MODEL_PATH}")
    print("[*] You need to either:")
    print("    1. Train a ResNet-50 model on TinyImageNet, or")
    print("    2. Use a pre-trained model and place it at the expected location")
    print(f"    Expected location: {MODEL_PATH}")
    
    # Option to train (commented out as it takes a long time)
    train_model = input("\nDo you want to train a new model? This will take several hours. (y/N): ")
    if train_model.lower() == 'y':
        print("[*] Training ResNet-50 on TinyImageNet...")
        
        # Create environment with CUDA_VISIBLE_DEVICES
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1
        
        subprocess.run([
            "python", "src/classification/main_train.py",
            "--arch", ARCH,
            "--dataset", DATASET,
            "--imagenet_arch",  # Important: Use ImageNet architecture for 64x64 images
            "--pretrained",     # Use pretrained ImageNet weights
            "--lr", "0.001",    # Even lower learning rate for fine-tuning
            "--epochs", "100",  # Adjust as needed
            "--save_dir", RESULTS_DIR,
            "--data_dir", TIN_IMAGENET_DIR,
            "--batch_size", "64"  # Reduce batch size to avoid OOM
        ], check=True, env=env)
        print("[*] Training complete. Please check the results directory for the trained model.")
        print(f"[*] Move the best model to: {MODEL_PATH}")
    else:
        print("[*] Please provide a trained ResNet-50 model and place it at the expected location.")
        sys.exit(1)

# Load the forget mask to get the number of samples to forget
try:
    forget_mask = torch.load(FORGET_MASK_PATH)
    num_to_forget = forget_mask.sum().item()  # Count True values in boolean mask
    total_samples = len(forget_mask)
    print(f"[✓] Loaded boolean forget mask: {num_to_forget} samples to forget out of {total_samples} total")
except Exception as e:
    print(f"[!] Error loading forget mask: {e}")
    sys.exit(1)

# Validate that the TinyImageNet dataset exists
if not os.path.exists(TIN_IMAGENET_DIR):
    print(f"[!] TinyImageNet dataset not found at: {TIN_IMAGENET_DIR}")
    print("[*] Please download and extract TinyImageNet to the datasets folder")
    sys.exit(1)

# === Step 3: Generate saliency mask ===
print("\n[*] Generating saliency mask using generate_mask.py...")

# Create environment with CUDA_VISIBLE_DEVICES
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1

subprocess.run([
    "python", os.path.join(current_dir, "src/classification/generate_mask.py"),
    "--save_dir", SALIENCY_DIR,
    "--model_path", MODEL_PATH,
    "--subset_indices_path", FORGET_MASK_PATH,
    "--num_indexes_to_replace", str(num_to_forget),
    "--unlearn_epochs", "1",
    "--arch", ARCH,
    "--dataset", DATASET,
    "--imagenet_arch",  # Important: Use ImageNet architecture to match trained model
    "--train_y_file", os.path.join(current_dir, "labels_tinyimagenet", "train_ys.pth"),
    "--val_y_file", os.path.join(current_dir, "labels_tinyimagenet", "val_ys.pth")
], check=True, env=env)

# === Step 4: Run SalUn unlearning ===
print("\n[*] Running SalUn unlearning using main_random.py...")

# Create environment with CUDA_VISIBLE_DEVICES
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1

subprocess.run([
    "python", os.path.join(current_dir, "src/classification/main_random.py"),
    "--unlearn", "RL",  # CORRECT: Use RL (Random Labels) for SalUn as per official docs
    "--unlearn_epochs", "10",  # SalUn standard: 10 epochs with very low LR
    "--unlearn_lr", "0.005",  # SalUn-style: much lower LR
    "--num_indexes_to_replace", str(num_to_forget),
    "--model_path", MODEL_PATH,
    "--save_dir", SAVE_DIR,
    "--mask_path", MASK_PATH,
    "--subset_indices_path", FORGET_MASK_PATH,
    "--arch", ARCH,
    "--dataset", DATASET,
    "--imagenet_arch",  # Important: Use ImageNet architecture to match trained model
    "--train_y_file", os.path.join(current_dir, "labels_tinyimagenet", "train_ys.pth"),
    "--val_y_file", os.path.join(current_dir, "labels_tinyimagenet", "val_ys.pth")
], check=True, env=env)

print("\n[✓] Dog-class forgetting complete using SalUn + ResNet-50.")
