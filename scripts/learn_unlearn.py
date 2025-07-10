import subprocess
import os
import shutil

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(project_root)

saliency_dir = "masks/"
results_dir = "results/"
save_dir = "models/"

os.makedirs(saliency_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(results_dir, "0model_SA_best.pth.tar")
mask_path = os.path.join(saliency_dir, "mask.pth")
unlearned_model_path = os.path.join(save_dir, "unlearned_model.pth.tar")



print("\nTraining ResNet-18 on CIFAR-10...")
subprocess.run([
    "python", "src/classification/main_train.py",
    "--arch", "resnet18",
    "--dataset", "cifar10",
    "--lr", "0.1",
    "--epochs", "182",
    "--save_dir", results_dir
], check=True)

print("\nGenerating saliency map...")
subprocess.run([
    "python", "src/classification/generate_mask.py",
    "--save_dir", saliency_dir,
    "--model_path", model_path,
    "--num_indexes_to_replace", "4500",
    "--unlearn_epochs", "1"
], check=True)

print("\nRunning SalUN unlearning...")
subprocess.run([
    "python", "src/classification/main_random.py",
    "--unlearn", "RL",
    "--unlearn_epochs", "10",
    "--unlearn_lr", "0.013",
    "--num_indexes_to_replace", "4500",
    "--model_forget", model_path,
    "--save_dir", save_dir,
    "--mask_path", mask_path,
], check=True)

# Rename output model (if it overwrote original)
if os.path.exists(model_path):
    shutil.move(model_path, unlearned_model_path)
    print(f"\nUnlearned model saved as: {unlearned_model_path}")
else:
    print("\nWarning: Expected output model not found.")

print("\nDone.")
