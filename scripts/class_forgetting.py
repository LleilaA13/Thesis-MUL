import subprocess
import os
import shutil

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(project_root)

forget_class = 3  # Example: forget class 3

saliency_dir = f"masks/class_forgetting/"
results_dir = f"results/class_forgetting/"
save_dir = f"models/class_forgetting/"

os.makedirs(saliency_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(results_dir, "0model_SA_best.pth.tar")
mask_path = os.path.join(saliency_dir, "with_0.5.pt")  # or another threshold
unlearned_model_path = os.path.join(
    save_dir, f"unlearned_model_class_{forget_class}.pth.tar")
"""
print("\nTraining ResNet-18 on CIFAR-10...")
subprocess.run([
    "python", "src/classification/main_train.py",
    "--arch", "resnet18",
    "--dataset", "cifar10",
    "--lr", "0.1",
    "--epochs", "182",
    "--save_dir", results_dir
], check=True)
"""
print("\nGenerating saliency map for class forgetting...")
subprocess.run([
    "python", "src/classification/generate_mask.py",
    "--save_dir", saliency_dir,
    "--model_path", model_path,
    "--class_to_replace", str(forget_class),
    "--num_indexes_to_replace", "4500",
    "--unlearn_epochs", "1"
], check=True)

print("\nRunning class-based unlearning...")
subprocess.run([
    "python", "src/classification/main_random.py",
    "--unlearn", "RL",
    "--unlearn_epochs", "10",
    "--unlearn_lr", "0.013",
    "--num_indexes_to_replace", "4500",
    "--model_path", model_path,  # <-- change here
    "--save_dir", save_dir,
    "--mask_path", mask_path,
    "--class_to_replace", str(forget_class)
], check=True)

print("\nDone.")
