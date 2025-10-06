import subprocess
import os
import sys
import torch
from tqdm import tqdm

# === Setup project paths ===
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "src", "classification"))

# === Config ===
# Example: Forgetting 'plain' and 'ecc' classes (replace with actual class indices)
PLAIN_ECC_CLASS_IDS = [class_id_plain, class_id_ecc]  # Fill in with correct indices
DATASET = "tinyimagenet"
ARCH = "resnet50"
TIN_IMAGENET_DIR = os.path.join(current_dir, "datasets", "tiny-imagenet-200")
MODEL_PATH = os.path.join(current_dir, "src/classification/resnet50.pth")
FORGET_MASK_PATH = os.path.join(current_dir, "plain_ecc_forget_indices.pt")
SALIENCY_DIR = os.path.join(current_dir, "masks/resnet50_plain_ecc_forgetting")
SAVE_DIR = os.path.join(current_dir, "models/resnet50_plain_ecc_forgetting/mask0_5")
MASK_PATH = os.path.join(SALIENCY_DIR, "with_0.5.pt")  # or another threshold

os.makedirs(SALIENCY_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# === Step 1: Prepare Forget Mask ===
# You need to create a mask file (plain_ecc_forget_indices.pt) with indices of samples to forget.
# This can be done with a helper script or notebook that selects all samples from the target classes.
num_to_forget = int(torch.load(FORGET_MASK_PATH).sum().item())

# === Step 2: Generate saliency mask ===
print("\n[*] Generating saliency mask using generate_mask.py...")
subprocess.run([
    "python", os.path.join(current_dir, "src/classification/generate_mask.py"),
    "--save_dir", SALIENCY_DIR,
    "--model_path", MODEL_PATH,
    "--subset_indices_path", FORGET_MASK_PATH,
    "--num_indexes_to_replace", str(num_to_forget),
    "--unlearn_epochs", "1",
    "--arch", ARCH,
    "--dataset", DATASET,
    "--train_y_file", os.path.join(current_dir, "labels", "train_ys.pth"),
    "--val_y_file", os.path.join(current_dir, "labels", "val_ys.pth")
], check=True)

# === Step 3: Run SalUn unlearning ===
print("\n[*] Running SalUn unlearning using main_random.py...")
subprocess.run([
    "python", os.path.join(current_dir, "src/classification/main_random.py"),
    "--unlearn", "RL",  # or your chosen method
    "--unlearn_epochs", "1",
    "--unlearn_lr", "0.01",
    "--num_indexes_to_replace", str(num_to_forget),
    "--model_path", MODEL_PATH,
    "--save_dir", SAVE_DIR,
    "--mask_path", MASK_PATH,
    "--subset_indices_path", FORGET_MASK_PATH,
    "--arch", ARCH,
    "--dataset", DATASET,
    "--train_y_file", os.path.join(current_dir, "labels", "train_ys.pth"),
    "--val_y_file", os.path.join(current_dir, "labels", "val_ys.pth")
], check=True)

print("\n[✓] Plain/ECC-class forgetting complete using SalUn + ResNet-50.")
