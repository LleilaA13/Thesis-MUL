# Random Data Forgetting + Feature Visualization Research

This repository implements a comprehensive analysis pipeline for studying how **random data forgetting** affects learned features in neural networks, with visualization using **Lucent**.

## 🎯 Research Objectives

1. **Analyze Random Data Forgetting**: Study how randomly forgetting different percentages of training data affects model performance
2. **Weight Influence Analysis**: Identify which specific weights and layers are most affected by random forgetting
3. **Feature Visualization**: Use Lucent to visualize how learned features change after random data forgetting
4. **Comparative Study**: Compare different unlearning methods (Random Labels vs Gradient Ascent) and mask thresholds

## 🏗️ Architecture Overview

```
Random Data Forgetting Pipeline
├── Baseline Training (Clean Model)
├── Random Index Generation (10%, 20%, 30% forgetting)
├── Saliency Mask Generation (Thresholds: 0.3, 0.5, 0.7)
├── Unlearning Experiments (RL + GA methods)
├── Weight Change Analysis
└── Feature Visualization (Lucent)
```

## 📁 Repository Structure

```
experiments/random_forgetting/
├── models/                     # Trained models
│   ├── baseline/              # Original trained model
│   └── ratio_X_mask_Y_method_Z/  # Unlearned models
├── masks/                     # Saliency masks
├── visualizations/            # Feature visualizations
│   ├── features/             # Lucent feature maps
│   └── weight_changes/       # Weight analysis plots
├── weight_analysis/           # Weight change analysis
└── forget_indices_*.npy      # Random forgetting indices
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone and navigate to the repository
cd Unlearn-Saliency

# Install requirements
pip install -r requirements_research.txt

# Make scripts executable
chmod +x run_research_pipeline.sh
```

### 2. Run Complete Pipeline

```bash
# Run the complete research pipeline
./run_research_pipeline.sh
```

Choose option 1 for the complete pipeline, which will:
- Train baseline ResNet-18 on CIFAR-10
- Generate random forgetting indices (10%, 20%, 30%)
- Create saliency masks with different thresholds
- Run unlearning experiments with RL and GA methods
- Analyze weight changes
- Generate feature visualizations

### 3. Quick Test Run

For a quick test with reduced epochs:

```bash
./run_research_pipeline.sh
# Choose option 4 for quick test
```

## 📊 Analysis Components

### 1. Weight Influence Analysis (`weight_influence_analyzer.py`)

Analyzes which weights are most affected by random forgetting:

- **Layer Sensitivity**: Which layers change most during forgetting
- **Weight Distribution Changes**: How weight distributions shift
- **Individual Weight Tracking**: Most affected individual weights
- **Statistical Analysis**: KL divergence, Wasserstein distance

### 2. Feature Visualization (`lucent_visualizer.py`)

Uses Lucent to visualize feature changes:

- **Before/After Comparisons**: Feature maps before and after unlearning
- **Layer-wise Analysis**: Focus on key layers (early, middle, late)
- **Channel Visualization**: Individual channel feature maps
- **Input Attribution**: How specific inputs affect different layers

### 3. Research Pipeline (`research_pipeline.py`)

Orchestrates the complete experimental workflow:

- **Baseline Training**: Train clean model for comparison
- **Random Index Generation**: Create random forgetting sets
- **Saliency Mask Generation**: Create weight importance masks
- **Unlearning Experiments**: Run systematic forgetting experiments
- **Result Aggregation**: Collect and organize all results

## 🔬 Experimental Parameters

### Default Configuration

```python
config = {
    'arch': 'resnet18',           # Model architecture
    'dataset': 'cifar10',         # Dataset
    'train_epochs': 100,          # Baseline training epochs
    'train_lr': 0.1,             # Training learning rate
    'unlearn_epochs': 10,         # Unlearning epochs
    'unlearn_lr': 0.013,         # Unlearning learning rate (from SalUn paper)
    'batch_size': 128             # Batch size
}
```

### Experimental Matrix

- **Forget Ratios**: 10%, 20%, 30% of training data
- **Mask Thresholds**: 0.3, 0.5, 0.7 (blocks 70%, 50%, 30% of weights)
- **Unlearn Methods**: RL (Random Labels), GA (Gradient Ascent)
- **Total Experiments**: 3 × 3 × 2 = 18 experiments

## 📈 Key Research Questions

1. **Which layers are most sensitive to random data forgetting?**
   - Early layers (low-level features) vs late layers (high-level features)
   - Convolutional vs fully connected layers

2. **How do different unlearning methods affect features?**
   - Random Labels (RL) vs Gradient Ascent (GA)
   - Which method preserves useful features better?

3. **What is the effect of saliency mask threshold?**
   - More restrictive masks (0.3) vs less restrictive (0.7)
   - Trade-off between forgetting effectiveness and feature preservation

4. **How do features visually change after forgetting?**
   - Using Lucent to visualize feature maps
   - Comparing feature complexity before/after

## 🛠️ Manual Execution

If you prefer to run components individually:

### Train Baseline Model

```bash
python Classification/main_train.py \
    --arch resnet18 \
    --dataset cifar10 \
    --epochs 100 \
    --save_dir experiments/random_forgetting/models/baseline
```

### Generate Saliency Masks

```bash
python Classification/generate_mask.py \
    --arch resnet18 \
    --dataset cifar10 \
    --model_path experiments/random_forgetting/models/baseline/model_best.pth.tar \
    --save_dir experiments/random_forgetting/masks/ratio_0.1 \
    --num_indexes_to_replace 5000
```

### Run Unlearning

```bash
python Classification/main_random.py \
    --unlearn RL \
    --unlearn_epochs 10 \
    --unlearn_lr 0.013 \
    --num_indexes_to_replace 5000 \
    --model_path experiments/random_forgetting/models/baseline/model_best.pth.tar \
    --save_dir experiments/random_forgetting/models/ratio_0.1_mask_0.5_method_RL \
    --mask_path experiments/random_forgetting/masks/ratio_0.1/with_0.5.pt
```

### Analyze Results

```bash
python weight_influence_analyzer.py
python lucent_visualizer.py
```

## 📊 Expected Results

### Weight Analysis Output

- `comprehensive_weight_analysis.json`: Detailed numerical analysis
- `summary_report.md`: Human-readable summary
- `layer_changes_*.png`: Visualization of layer-wise changes

### Feature Visualization Output

- `feature_comparison_*.png`: Before/after feature comparisons
- `attribution_*.png`: Input attribution analysis

## 🔧 Customization

### Change Dataset/Architecture

Edit the config in `research_pipeline.py`:

```python
config = {
    'arch': 'resnet50',          # Change to resnet50
    'dataset': 'TinyImagenet',   # Change to TinyImageNet
    # ... other parameters
}
```

### Add New Analysis

Extend the analyzers:

```python
class CustomAnalyzer:
    def custom_analysis(self):
        # Your custom analysis code
        pass
```

## 🧪 For Your Thesis

### Key Findings to Look For

1. **Layer Sensitivity Patterns**: Document which layers change most
2. **Method Comparison**: Compare RL vs GA effectiveness
3. **Feature Preservation**: How well are useful features preserved?
4. **Threshold Effects**: Impact of different mask thresholds

### Suggested Visualizations

1. **Heatmaps**: Layer sensitivity across experiments
2. **Feature Galleries**: Before/after feature comparisons
3. **Weight Distribution Plots**: How distributions shift
4. **Performance Metrics**: Forget accuracy vs retain accuracy

## 🤝 Contributing

This research pipeline is designed for academic research. Feel free to:

- Add new analysis methods
- Implement additional visualization techniques
- Extend to new architectures/datasets
- Improve the experimental design

## 📚 References

- **SalUn Paper**: [Empowering Machine Unlearning via Gradient-based Weight Saliency](https://arxiv.org/abs/2310.12508)
- **Lucent**: [Feature Visualization Library](https://github.com/greentfrapp/lucent)
- **Original SalUn Repository**: [OPTML-Group/Unlearn-Saliency](https://github.com/OPTML-Group/Unlearn-Saliency)

## 📄 License

This project is licensed under the same terms as the original SalUn repository.

---

**Happy Researching! 🔬✨**