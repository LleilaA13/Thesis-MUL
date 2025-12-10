# Extracting Forgotten Images for Feature Visualization

This guide explains how to get the actual images that were "forgotten" during unlearning, which you can then use in your Lucent feature visualization notebook.

## Step 1: Run the Unlearning Experiment

When you run your unlearning experiment, the script will now automatically save the forgotten indices:

```bash
conda activate salUN

cd /media/hdd/usr/leyla/Unlearn-Saliency/core/Classification

python main_random.py --unlearn RL --dataset TinyImagenet --arch resnet50 --imagenet_arch \
  --data_dir ../../datasets/tiny-imagenet-200 \
  --unlearn_epochs 15 --unlearn_lr 0.005 \
  --num_indexes_to_replace 10000 \
  --model_path ../../experiments/models/resnet50_pretrained.pth \
  --save_dir ../../experiments/results/good_results/random_forgetting_10percent_RL_tweak_conservative \
  --mask_path ../../experiments/masks/random_forgetting_10percent/with_0.6.pt
```

After running, you'll find `forgotten_indices.pt` in your save directory.

## Step 2: Extract the Forgotten Images

Use the extraction script to copy the forgotten images to a separate folder:

```bash
cd /media/hdd/usr/leyla/Unlearn-Saliency

python scripts/extract_forgotten_data.py \
  --indices_path experiments/results/good_results/random_forgetting_10percent_RL_tweak_conservative/forgotten_indices.pt \
  --data_dir datasets/tiny-imagenet-200/train \
  --output_dir forgotten_images_10percent
```

This will create a folder structure like:
```
forgotten_images_10percent/
├── n01443537/
│   ├── image1.JPEG
│   ├── image2.JPEG
├── n01629819/
│   ├── image3.JPEG
└── ...
```

## Step 3: Use in Your Notebook

Now you can load these forgotten images in your Lucent notebook for visualization:

```python
from PIL import Image
from torchvision import transforms
import os

# Path to forgotten images
forgotten_dir = "forgotten_images_10percent"

# Load a forgotten image
forgotten_images = []
for class_name in os.listdir(forgotten_dir):
    class_path = os.path.join(forgotten_dir, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = Image.open(img_path)
        forgotten_images.append((img, class_name, img_name))

print(f"Loaded {len(forgotten_images)} forgotten images")

# Example: Visualize how the model reacts to a forgotten image
forgotten_img, class_name, img_name = forgotten_images[0]

# Convert to tensor
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])
img_tensor = transform(forgotten_img).unsqueeze(0).to(device)

# Get activations from pretrained vs unlearned model
with torch.no_grad():
    activations_pretrained = pretrained_model(img_tensor)
    activations_unlearned = unlearned_model(img_tensor)

# Compare predictions
pred_pretrained = torch.argmax(activations_pretrained)
pred_unlearned = torch.argmax(activations_unlearned)

print(f"Pretrained model prediction: {pred_pretrained}")
print(f"Unlearned model prediction: {pred_unlearned}")
```

## Troubleshooting

- **If indices file is missing**: Re-run the unlearning experiment with the updated `main_random.py`.
- **If indices seem wrong**: Check that the mask file and dataset match the experiment you're analyzing.
- **For different forgetting ratios**: Run the extraction for each experiment's saved indices separately.

## Next Steps for Feature Visualization

1. **Compare model responses**: Feed forgotten images to both models and compare:
   - Final predictions
   - Layer activations
   - Attention/saliency maps

2. **Feature inversion on forgotten images**: Use Lucent's feature inversion to reconstruct what each model "sees" in the forgotten images.

3. **Activation differences**: Compute activation deltas (like the "scar" visualization) but using real forgotten images instead of generated ones.
