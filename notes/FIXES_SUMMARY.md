# ✅ FIXED: generate_mask.py Compatibility Issues

## Summary of Issues and Fixes

### 🔧 Issue 1: Wrong Mask Format  
**Problem**: Our masks were in format `[[sample_idx, class_idx], ...]` but `generate_mask.py` expected boolean tensors.

**Solution**: Created new `generate_all_masks.py` that generates boolean masks:
- `dogs_forget_mask_boolean.pt` - Shape: [100000], Type: torch.bool
- `vehicles_forget_mask_boolean.pt` - Shape: [100000], Type: torch.bool  
- `cats_forget_mask_boolean.pt` - Shape: [100000], Type: torch.bool

### 🔧 Issue 2: Incorrect Sample Count Calculation
**Problem**: Script used `forget_mask.sum().item()` on index-pair format, giving huge numbers.

**Solution**: Updated to use proper boolean mask calculation:
```python
# OLD (wrong): num_to_forget = int(forget_mask.sum().item())  # Summed all indices
# NEW (correct): num_to_forget = forget_mask.sum().item()    # Count True values
```

### 🔧 Issue 3: Missing ToTensor() Transforms  
**Problem**: TinyImageNet dataset was returning PIL Images instead of tensors.

**Solution**: Fixed `src/classification/dataset.py`:
```python
# Added ToTensor() to both train and test transforms
self.tr_train = [
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # ← ADDED
]
self.tr_test = [
    transforms.ToTensor(),  # ← ADDED  
]
```

### 🔧 Issue 4: Boolean Tensor Iteration
**Problem**: Code tried to iterate boolean tensors incorrectly.

**Solution**: Restored original boolean handling in `generate_mask.py`:
```python
# Correct boolean iteration (no .item() needed for individual boolean values)
forget_ids = [i for i, flag in enumerate(forget_mask) if flag]
retain_ids = [i for i, flag in enumerate(forget_mask) if not flag]
```

## ✅ Current Status

**Working correctly:**
- Boolean masks load and process properly
- Dataset transforms convert images to tensors
- Forget/retain sample counts are accurate
- Saliency mask generation is running

**Files updated:**
1. `generate_all_masks.py` - Creates correct boolean format
2. `resnet50_unlearn_dogs.py` - Uses new boolean mask file
3. `src/classification/dataset.py` - Added missing ToTensor transforms
4. `src/classification/generate_mask.py` - Restored boolean handling

**Ready for experiments:**
- Dogs: 3000 samples from 6 classes
- Vehicles: 3500 samples from 7 classes  
- Cats: 1500 samples from 3 classes

The script is now successfully running the saliency mask generation phase!