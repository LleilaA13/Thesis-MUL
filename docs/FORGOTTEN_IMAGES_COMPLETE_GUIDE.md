# Complete Guide: Working with Forgotten Images

## Quick Start - Already Have Experiment Results?

If you have existing unlearning results (e.g., `random_forgetting_10percent_RL_tweak_conservative`), follow these 3 steps:

### 1. Reconstruct Forgotten Indices
```bash
python scripts/reconstruct_forgotten_indices.py \
    --data_dir datasets/tiny-imagenet-200 \
    --num_indexes_to_replace 10000 \
    --seed 1 \
    --output_path experiments/results/good_results/random_forgetting_10percent_RL_tweak_conservative/forgotten_indices.pt
```

**Output**: `forgotten_indices.pt` containing 10,000 sample indices

### 2. Extract Forgotten Images
```bash
python scripts/extract_forgotten_data.py \
    --indices_path experiments/results/good_results/random_forgetting_10percent_RL_tweak_conservative/forgotten_indices.pt \
    --data_dir datasets/tiny-imagenet-200/train \
    --output_dir experiments/results/good_results/random_forgetting_10percent_RL_tweak_conservative/forgotten_images
```

**Output**: Directory with 10,000 extracted JPEG images

### 3. Visualize Samples (Optional)
```bash
python scripts/visualize_forgotten_images.py \
    --forgotten_dir experiments/results/good_results/random_forgetting_10percent_RL_tweak_conservative/forgotten_images/images \
    --num_images 16
```

**Output**: `forgotten_samples_visualization.png` with 16 random samples

---

## Understanding the Scripts

### `reconstruct_forgotten_indices.py`
Recreates which samples were forgotten by:
- Loading the dataset with same seed
- Applying same random selection logic (class_to_replace=-1 for random forgetting)
- Extracting indices where targets < 0 (marked as forgotten)

**Key Parameters:**
- `--num_indexes_to_replace`: How many samples were forgotten (e.g., 10000 for 10%)
- `--seed`: Must match the seed used in original experiment
- `--data_dir`: Path to Tiny ImageNet dataset

### `extract_forgotten_data.py`
Copies actual image files based on indices:
- Loads `forgotten_indices.pt`
- Uses ImageFolder to access dataset
- Copies images preserving class structure

### `visualize_forgotten_images.py`
Creates a visualization grid of random samples for quick verification

---

## For New Experiments

### Option A: Updated main_random.py (Automatic)

The current `main_random.py` automatically saves forgotten indices:

```bash
cd core/Classification
python main_random.py \
    --arch resnet50 \
    --dataset TinyImagenet \
    --data datasets/tiny-imagenet-200 \
    --num_indexes_to_replace 10000 \
    --save_dir ../../experiments/results/my_experiment \
    --unlearn RL \
    --mask ../../experiments/masks/my_mask.pt
```

Then just run step 2 (extract images) and step 3 (visualize).

### Option B: Reconstruct Later

Run the experiment first, then use the reconstruction script as shown in Quick Start.

---

## Using Forgotten Images in Lucent

### Loading Images
```python
from PIL import Image
import os
from pathlib import Path

# Path to forgotten images
forgotten_dir = Path('experiments/results/good_results/random_forgetting_10percent_RL_tweak_conservative/forgotten_images/images')

# Load all images
forgotten_images = []
for img_file in sorted(forgotten_dir.glob('*.JPEG')):
    img = Image.open(img_file)
    forgotten_images.append(img)

print(f"Loaded {len(forgotten_images)} forgotten images")
```

### Prepare for Model Input
```python
import torch
from torchvision import transforms

# Tiny ImageNet normalization
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.Resize(64),  # Tiny ImageNet size
    transforms.ToTensor(),
    normalize
])

# Convert to tensor batch
image_tensors = [transform(img) for img in forgotten_images[:100]]
batch = torch.stack(image_tensors)
```

### Compare Model Responses
```python
import torch.nn.functional as F

# Load your models
pretrained_model = ...  # Load from models/resnet50/pretrained.pth
unlearned_model = ...   # Load from experiments/.../RLcheckpoint.pth.tar

pretrained_model.eval()
unlearned_model.eval()

with torch.no_grad():
    # Get predictions
    pretrained_outputs = pretrained_model(batch)
    unlearned_outputs = unlearned_model(batch)
    
    # Compare confidence/predictions
    pretrained_probs = F.softmax(pretrained_outputs, dim=1)
    unlearned_probs = F.softmax(unlearned_outputs, dim=1)
    
    # Analyze differences
    confidence_drop = (pretrained_probs.max(1)[0] - unlearned_probs.max(1)[0]).mean()
    print(f"Average confidence drop: {confidence_drop:.4f}")
```

### Feature Visualization with Lucent
```python
from lucent.optvis import render, param, transform
from lucent.modelzoo import inceptionv1

# Use forgotten images as optimization targets
for img_idx in range(10):
    img_path = forgotten_images[img_idx]
    
    # Visualize what neurons activate for this forgotten image
    _ = render.render_vis(
        pretrained_model,
        f"layer4:0",  # Choose your layer
        image=img_path,
        thresholds=(256,),
        show_inline=True
    )
```

---

## Directory Structure After Extraction

```
experiments/results/good_results/random_forgetting_10percent_RL_tweak_conservative/
├── forgotten_indices.pt                    # 10,000 sample indices
├── forgotten_images/                       # Extracted images
│   └── images/
│       ├── n01443537_103.JPEG
│       ├── n01443537_106.JPEG
│       └── ... (10,000 total)
├── forgotten_samples_visualization.png     # Sample grid
├── FORGOTTEN_DATA_README.md               # This directory's documentation
├── RLcheckpoint.pth.tar                   # Unlearned model
├── RLeval_result.pth.tar                  # Evaluation results
└── with_0.6.pt                            # Saliency mask
```

---

## Troubleshooting

### "Found 0 forgotten sample indices"
- Check that `--seed` matches original experiment
- Verify `--num_indexes_to_replace` is correct
- Ensure `class_to_replace=-1` for random forgetting (handled automatically in script)

### "AttributeError: module X has no attribute Y"
- Run from repository root: `cd /path/to/Unlearn-Saliency`
- Activate conda environment: `conda activate salUN`
- Check Python path includes `core/Classification`

### Images look wrong
- Verify dataset path points to `train/` directory
- Check that dataset hasn't been modified since experiment

---

## Parameters for Different Experiments

### Random 10% Forgetting
```bash
--num_indexes_to_replace 10000
--seed 1
```

### Random 20% Forgetting
```bash
--num_indexes_to_replace 20000
--seed 1
```

### Random 30% Forgetting
```bash
--num_indexes_to_replace 30000
--seed 1
```

### Class-Specific Forgetting (e.g., class 5)
Note: For class forgetting, you need to know which class was forgotten. Check your experiment config.

---

## Next Steps for Thesis

1. **Quantitative Analysis**: Measure prediction changes on forgotten images
2. **Feature Visualization**: Generate Lucent activations for key layers
3. **Comparison Plots**: Side-by-side pretrained vs unlearned responses
4. **Weight Correlation**: Link to `good_results_weight_analysis` findings
5. **Thesis Integration**: Use visualizations in results chapter

## Related Documentation

- `docs/METHODOLOGY.md`: Full experimental methodology
- `analysis/weight_analysis/good_results_weight_analysis/README.md`: Weight analysis results
- `notebooks/README.md`: Lucent notebook usage guide
