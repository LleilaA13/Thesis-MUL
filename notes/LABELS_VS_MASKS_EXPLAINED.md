# Labels vs Forget Masks: Understanding the Difference

## 🔑 KEY CONCEPT

**Labels and forget masks serve completely different purposes:**

### 📊 LABELS (train_ys.pth, val_ys.pth)
- **Purpose**: Tell the model the TRUE CLASS of each sample
- **Same for ALL experiments** (dogs, vehicles, cats)
- **Content**: Class indices (0-199 for TinyImageNet)
- **Usage**: Model training, evaluation, loss calculation

### 🎯 FORGET MASKS (*_forget_indices.pt)  
- **Purpose**: Specify WHICH SAMPLES to forget during unlearning
- **Different for each experiment**
- **Content**: Sample indices + their class labels
- **Usage**: Saliency mask generation, unlearning process

## 📁 OUR SETUP (CORRECT)

### Labels (Same for All Tasks):
```
labels_tinyimagenet/
├── train_ys.pth    # 100,000 samples, classes 0-199
└── val_ys.pth      # 10,000 samples, classes 0-199
```

### Forget Masks (Different for Each Task):
```
dogs_forget_indices.pt           # 3,000 samples from 6 dog classes
vehicles_forget_indices.pt       # 3,500 samples from 7 vehicle classes  
cats_forget_indices_resnet50.pt  # 1,500 samples from 3 cat classes
```

## 🔍 HOW IT WORKS

### For Dog Forgetting Experiment:
1. **Labels**: Use `labels_tinyimagenet/` (tells model all 200 classes)
2. **Forget mask**: Use `dogs_forget_indices.pt` (marks dog samples for forgetting)
3. **Result**: Model forgets dogs but remembers vehicles, cats, and all other classes

### For Vehicle Forgetting Experiment:
1. **Labels**: Use `labels_tinyimagenet/` (same labels as dog experiment)
2. **Forget mask**: Use `vehicles_forget_indices.pt` (marks vehicle samples for forgetting)  
3. **Result**: Model forgets vehicles but remembers dogs, cats, and all other classes

### For Cat Forgetting Experiment:
1. **Labels**: Use `labels_tinyimagenet/` (same labels as other experiments)
2. **Forget mask**: Use `cats_forget_indices_resnet50.pt` (marks cat samples for forgetting)
3. **Result**: Model forgets cats but remembers dogs, vehicles, and all other classes

## ✅ WHY THIS IS CORRECT

- **Evaluation consistency**: We can compare results across experiments because the ground truth (labels) is the same
- **Fair comparison**: Each experiment starts with the same base knowledge (all 200 classes)
- **Proper unlearning**: Only the targeted class samples are affected, others remain intact

## ❌ WHAT WOULD BE WRONG

Creating different label files for each task would mean:
- Different ground truth for each experiment
- No fair comparison between results
- Potential evaluation errors
- Unnecessary complexity

## 🎯 SUMMARY

**✅ Current setup is PERFECT:**
- One set of labels for all experiments
- Different forget masks for each target class group
- Clean, comparable results across all unlearning tasks