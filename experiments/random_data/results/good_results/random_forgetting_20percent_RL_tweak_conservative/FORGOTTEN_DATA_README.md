# Forgotten Data Extraction - Random 20% Forgetting

## Overview
This directory contains the forgotten data from the **Random 20% Forgetting** experiment using the **RL (Re-Labeling) method with conservative tweaking**.

## Dataset Details
- **Total forgotten samples**: 20,000 images
- **Dataset**: Tiny ImageNet (200 classes)
- **Forgetting strategy**: Random selection across all classes
- **Random seed**: 1

## Directory Structure
```
forgotten_images/
└── images/
    ├── n01443537_103.JPEG
    ├── n01443537_106.JPEG
    ├── ...
    └── (20,000 total images)
```

## Files in This Directory
- `forgotten_indices.pt`: PyTorch tensor containing the 20,000 indices of forgotten samples
- `forgotten_images/`: Directory with extracted image files
- `RLcheckpoint.pth.tar`: Unlearning checkpoint
- `RLeval_result.pth.tar`: Evaluation results
- `with_0.6.pt`: Saliency mask at threshold 0.6

## Sample Index Statistics
- **Min index**: 3
- **Max index**: 99,993
- **First 10 indices**: [3, 6, 9, 14, 17, 23, 25, 26, 34, 36]

## How to Use the Forgotten Images

### In Python Scripts
```python
import torch
from torchvision import datasets, transforms

# Load indices
forgotten_indices = torch.load('forgotten_indices.pt')
print(f"Number of forgotten samples: {len(forgotten_indices)}")

# Load images directly
from torchvision.datasets import ImageFolder
forgotten_dataset = ImageFolder(
    'forgotten_images',
    transform=transforms.ToTensor()
)
```

### In Lucent Notebooks for Feature Visualization
```python
from PIL import Image
import os

# Load all forgotten images
forgotten_dir = 'experiments/results/good_results/random_forgetting_20percent_RL_tweak_conservative/forgotten_images/images'
forgotten_images = []

for img_name in os.listdir(forgotten_dir):
    img_path = os.path.join(forgotten_dir, img_name)
    img = Image.open(img_path)
    forgotten_images.append(img)

# Use with your pretrained and unlearned models
# Compare how each model responds to these forgotten images
```

## Regeneration Process
If you need to regenerate these indices:
```bash
python scripts/reconstruct_forgotten_indices.py \
    --data_dir datasets/tiny-imagenet-200 \
    --num_indexes_to_replace 20000 \
    --seed 1 \
    --output_path forgotten_indices.pt
```

Then extract images:
```bash
python scripts/extract_forgotten_data.py \
    --indices_path forgotten_indices.pt \
    --data_dir datasets/tiny-imagenet-200/train \
    --output_dir forgotten_images
```

## Next Steps for Analysis
1. **Feature Visualization**: Use Lucent to generate channel activations for forgotten images
2. **Comparison Analysis**: Compare pretrained vs unlearned model responses
3. **Weight Analysis**: Correlate with `good_results_weight_analysis` findings
4. **Thesis Integration**: Use visualizations in methodology/results sections

## Related Files
- Weight analysis: `analysis/weight_analysis/good_results_weight_analysis/`
- Unlearning script: `core/Classification/main_random.py`
- Extraction scripts: `scripts/reconstruct_forgotten_indices.py`, `scripts/extract_forgotten_data.py`
