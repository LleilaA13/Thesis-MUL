import subprocess
import os
import sys
import torch
from tqdm import tqdm

# === GPU Configuration ===
# Set which GPU to use (0 or 1, or "0,1" for both)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1 which has more free memory
print(f"[*] Using GPU(s): {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")

# === Setup project paths ===
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "src", "classification"))
sys.path.append(current_dir)  # Add for unlearn_config

# Import centralized configuration
from unlearn_config import get_forget_class_config, create_forget_mask

# === Config ===
FORGET_TYPE = "cats"  # What we're forgetting
config = get_forget_class_config(FORGET_TYPE)
print(f"[*] Configured for {FORGET_TYPE} forgetting: {len(config['wnids'])} classes")
print(f"[*] Classes: {config['names']}")

DATASET = "TinyImagenet"
ARCH = "resnet50"
TIN_IMAGENET_DIR = os.path.join(current_dir, "datasets/tiny-imagenet-200")
MODEL_PATH = os.path.join(current_dir, "src/classification/models/resnet50_pretrained.pth")
FORGET_MASK_PATH = os.path.join(current_dir, f"{FORGET_TYPE}_forget_mask_boolean.pt")
SALIENCY_DIR = os.path.join(current_dir, f"masks/resnet50_{FORGET_TYPE}_forgetting")
SAVE_DIR = os.path.join(current_dir, f"models/resnet50_{FORGET_TYPE}_forgetting/mask0_5_finetune_conservative")
RESULTS_DIR = os.path.join(current_dir, f"results/resnet50_{FORGET_TYPE}_forgetting")
MASK_PATH = os.path.join(SALIENCY_DIR, "with_0.5.pt")

os.makedirs(SALIENCY_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Step 1: Check or create the forget mask ===
if not os.path.exists(FORGET_MASK_PATH):
    print(f"\n[!] Forget mask not found at {FORGET_MASK_PATH}")
    print("[*] Creating cats forget mask...")
    
    # Create the forget mask using centralized config
    forget_mask = create_forget_mask(FORGET_TYPE, total_samples=100000, dataset_type='train')
    torch.save(forget_mask, FORGET_MASK_PATH)
    print(f"[✓] Cats forget mask created: {forget_mask.sum().item()} samples marked for forgetting")

# === Step 2: Train or load base model ===
if not os.path.exists(MODEL_PATH):
    print(f"\n[!] Model not found at {MODEL_PATH}")
    print("[*] Please provide a trained ResNet-50 model and place it at the expected location.")
    sys.exit(1)

# Load the forget mask to get the number of samples to forget
try:
    forget_mask = torch.load(FORGET_MASK_PATH)
    num_to_forget = int(forget_mask.sum().item())
    print(f"[✓] Loaded forget mask: {num_to_forget} samples to forget out of {len(forget_mask)} total")
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

# === Step 4: Run Fine-Tuning Unlearning ===
print("\n[*] Running Fine-Tuning unlearning using main_random.py...")

# Create environment with CUDA_VISIBLE_DEVICES
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1

subprocess.run([
    "python", os.path.join(current_dir, "src/classification/main_random.py"),
    "--unlearn", "FT",  # Fine-Tuning method (trains only on retain data)
    "--unlearn_epochs", "1",  # REDUCED: Much fewer epochs to prevent overfitting
    "--unlearn_lr", "0.00001",  # REDUCED: Much lower LR for gentle fine-tuning
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

print("\n[✓] Cat-class forgetting complete using Fine-Tuning + ResNet-50.")
print("[*] Cat-CONSERVATIVE-FINE-TUNING parameters applied:")
print(f"    - Method: Fine-Tuning (trains only on retain data)")
print(f"    - Learning rate: 0.00001 (ultra-conservative to prevent overfitting)")
print(f"    - Epochs: 1 (reduced to prevent 100% accuracy overfitting)")
print(f"    - Mask: 0.5 (blocks 50% of weights)")
print(f"    - COMPARISON: Conservative FT vs GA (21% forget, 53.73% retain)")
print(f"    - REASON: Original FT params (LR=0.001, epochs=10) caused overfitting")