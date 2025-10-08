# Next Steps: Mask Fixes and Feature Visualization Strategy

## Current Status Analysis

### Mask Files Available:
1. **vehicles_forget_indices.pt** (WRONG) - targets non-vehicle classes
2. **vehicles_forget_indices_CORRECT.pt** - targets actual vehicles (convertible, moving van, sports car)
3. **cat_forget_indices.pt** - large file (10MB), possibly corrupted or wrong format
4. **cats_forget_indices.pt** - small file (100KB), likely correct

### Model Directories:
- `models/resnet50_vehicles_forgetting/` - experiments with WRONG mask
- `models/resnet50_vehicles_CORRECT/` - experiment with correct vehicle mask (in progress)
- `models/inceptionv3_cat_forgetting/` - cat experiments for InceptionV3

## Immediate Actions Required

### 1. Clean Up Wrong Experiments
```bash
# Move or rename the wrong vehicle experiments
mv models/resnet50_vehicles_forgetting models/resnet50_vehicles_WRONG_MASK
```

### 2. Verify and Fix Cat Masks
The `cat_forget_indices.pt` (10MB) seems corrupted or wrong format. We should:
- Use `cats_forget_indices.pt` (which targets the right cat classes)
- Or recreate cat mask with proper format

### 3. Best Classes for Feature Visualization

Based on TinyImageNet analysis, the **BEST classes for interpretable feature visualization** are:

#### **Cats (Recommended for Lucent):**
- **Index 0**: n02124075 - Egyptian cat
- **Index 66**: n02123045 - tabby, tabby cat  
- **Index 131**: n02123394 - Persian cat

#### **Vehicles (Current focus):**
- **Index 64**: n03796401 - moving van
- **Index 117**: n04285008 - sports car, sport car
- **Index 157**: n03100240 - convertible

#### **Alternative Animals (highly interpretable):**
- **Index 22**: n01443537 - goldfish, Carassius auratus
- **Index 11**: n02106662 - German shepherd dog
- **Index 44**: n02509815 - red panda, bear cat
- **Index 102**: n02125311 - cougar, puma, mountain lion

### 4. Prioritized Experiment Plan

#### **Phase 1: Complete Current Experiments**
1. **Wait for resnet50_vehicles_CORRECT to finish** - should show proper vehicle forgetting
2. **Verify results** - expect forget accuracy to drop significantly on vehicles

#### **Phase 2: Cat Experiments for Lucent**
1. **Create/verify correct cat mask** targeting indices [0, 66, 131]
2. **Run cat forgetting experiments** with multiple sparsity levels
3. **Generate Lucent visualizations** on cat classes (most interpretable)

#### **Phase 3: Alternative Classes (if needed)**
1. **Dog experiments** - German shepherd (highly recognizable)
2. **Goldfish experiments** - simple, distinctive features
3. **Red panda experiments** - unique visual features

### 5. Why These Classes Are Best for Visualization

#### **Cats (RECOMMENDED):**
- ✅ **Highly distinctive features** (whiskers, ears, eyes, fur patterns)
- ✅ **Clear semantic meaning** 
- ✅ **Good for Lucent interpretation** (facial features, textures)
- ✅ **Three varieties** (Egyptian, tabby, Persian) for diversity

#### **Vehicles:**
- ✅ **Clear geometric shapes** (wheels, windows, body)
- ✅ **Distinctive parts** easy to interpret
- ✅ **Good contrast with animals** for comparison

#### **Why NOT other classes:**
- ❌ Abstract objects (e.g., "bannister") - hard to interpret
- ❌ Food items - less distinctive features
- ❌ Clothing - too generic features

## Implementation Commands

### Fix cat mask (if needed):
```bash
cd /media/hdd/usr/leyla/Unlearn-Saliency
python3 -c "
import torch
from torchvision.datasets import ImageFolder

# Load dataset to get correct indices
dataset = ImageFolder('datasets/tiny-imagenet-200/train')
class_to_idx = dataset.class_to_idx

# Cat classes in TinyImageNet
cat_classes = ['n02124075', 'n02123045', 'n02123394']  # Egyptian, tabby, Persian
cat_indices = [list(class_to_idx.keys()).index(cls) for cls in cat_classes]
print(f'Cat class indices: {cat_indices}')

# Create mask for 1500 samples (3 classes × 500 each)
mask = []
for cls_idx in cat_indices:
    for sample_idx in range(500):  # 500 samples per class
        mask.append([sample_idx + cls_idx * 500, cls_idx])

mask_tensor = torch.tensor(mask)
torch.save(mask_tensor, 'cats_forget_indices_VERIFIED.pt')
print(f'Saved verified cat mask: {len(mask_tensor)} samples')
"
```

### Start cat experiment:
```bash
python src/classification/main_random.py --config_path src/classification/config.json --device cuda --exp_name resnet50_cats_forgetting --mask cats_forget_indices_VERIFIED.pt --sparsity_ratio 0.3
```

## Expected Outcomes

1. **Vehicle experiments** should show low forget accuracy (~10-30%) and high UA (~70-90%)
2. **Cat experiments** should provide excellent Lucent visualizations with interpretable features
3. **Feature analysis** should reveal clear semantic differences between forgot and retained features

## Timeline
- **Day 1**: Complete vehicle experiments, verify results
- **Day 2**: Set up and run cat experiments  
- **Day 3**: Generate Lucent visualizations and analysis
- **Day 4**: Compare results and write thesis sections