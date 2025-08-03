import subprocess
import os
import sys
import torch
from tqdm import tqdm


# === Setup project paths ==
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "src", "classification"))

from imagenet import prepare_data, get_x_y_from_data_dict

# === Config ===
CAT_CLASS_IDS = [281, 282, 283, 284, 285]
DATASET = "imagenet_zeus"
ARCH = "inceptionv1"
IMAGENET_DIR = "/media/pinas/datasets/imagenet_zeus"

MODEL_PATH = os.path.join(current_dir, "models/inceptionv1_fulltrain/original_model.pth")
FORGET_MASK_PATH = os.path.join(current_dir, "cat_forget_indices.pt")
SALIENCY_DIR = os.path.join(current_dir, "masks/inceptionv1_cat_forgetting")
SAVE_DIR = os.path.join(current_dir, "models/inceptionv1_cat_forgetting")
MASK_PATH = os.path.join(SALIENCY_DIR, "with_0.3.pt")  # or another threshold

os.makedirs(SALIENCY_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# === Step 1: Generate forget mask for cat classes ===
if not os.path.exists(FORGET_MASK_PATH):
    print("[*] Generating forget mask for cat classes...")
    ys = []
    loaders = prepare_data(
        dataset=DATASET,
        batch_size=1,
        shuffle=False,
        data_path=IMAGENET_DIR
    )
    for data in tqdm(loaders["train"], desc="Scanning training labels"):
        _, y = get_x_y_from_data_dict(data, "cpu")
        ys.append(y.item())

    ys = torch.tensor(ys)
    mask = torch.zeros_like(ys)
    for i, label in enumerate(ys):
        if label in CAT_CLASS_IDS:
            mask[i] = 1

    torch.save(mask, FORGET_MASK_PATH)
    print(f"[✓] Saved forget mask for {int(mask.sum().item())} cat samples.")

num_to_forget = int(torch.load(FORGET_MASK_PATH).sum().item())

# === Step 2: Retrain model from scratch on full ImageNet ===
print("\n[*] Training model from scratch on all classes...")
subprocess.run([
     "python", os.path.join(current_dir, "src/classification/main_train.py"),
    "--dataset", DATASET,
    "--arch", ARCH,
    "--epochs", "90",
    "--lr", "0.01",
    "--batch_size", "128",
    "--save_dir", os.path.join(current_dir, "models/inceptionv1_fulltrain"),
    "--train_y_file", os.path.join(current_dir,"labels", "train_ys.pth"),
    "--val_y_file", os.path.join(current_dir,"labels", "val_ys.pth"),
    "--subset_indices_path", FORGET_MASK_PATH,
], check=True)




# === Step 3: Generate saliency mask ===
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
    "--train_y_file", os.path.join(current_dir,"labels", "train_ys.pth"),
    "--val_y_file", os.path.join(current_dir,"labels", "val_ys.pth")
], check=True)

# === Step 4: Run SalUn unlearning ===
print("\n[*] Running SalUn unlearning using main_random.py...")
subprocess.run([
    "python", os.path.join(current_dir, "src/classification/main_random.py"),
    "--unlearn", "RL",
    "--unlearn_epochs", "5",
    "--unlearn_lr", "0.01",
    "--num_indexes_to_replace", str(num_to_forget),
    "--model_path", MODEL_PATH,
    "--save_dir", SAVE_DIR,
    "--mask_path", MASK_PATH,
    "--subset_indices_path", FORGET_MASK_PATH,
    "--arch", ARCH,
    "--dataset", DATASET,
    "--train_y_file", os.path.join(current_dir,"labels", "train_ys.pth"),
    "--val_y_file", os.path.join(current_dir,"labels", "val_ys.pth")
], check=True)

print("\n[✓] Cat-class forgetting complete using SalUn + InceptionV1.")

