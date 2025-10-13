# Weight Influence Analysis - Usage Guide

## 🎯 What This Does
Analyzes how your random data forgetting experiments have affected the neural network weights compared to the baseline pretrained model.

## 📁 Your Current Setup
Based on your directory structure:

```
/media/hdd/usr/leyla/Unlearn-Saliency/
├── models/
│   └── resnet50_pretrained.pth          # Your baseline model
├── results/                             # Your unlearning experiments
│   ├── random_forgetting_10percent_GA_fixed/
│   ├── random_forgetting_10percent_GA_original/
│   ├── random_forgetting_10percent_GA_strong/
│   ├── random_forgetting_10percent_RL/
│   ├── random_forgetting_10percent_RL_aggressive/
│   ├── random_forgetting_10percent_RL_conservative/
│   ├── random_forgetting_20percent_RL_0.5/
│   ├── random_forgetting_20percent_RL_aggressive/
│   ├── random_forgetting_20percent_RL_conservative/
│   ├── random_forgetting_20percent_RL_more_conservative/
│   ├── random_forgetting_20percent_RL_tweak_conservative/
│   ├── random_forgetting_30percent_RL_conservative/
│   └── random_forgetting_30percent_RL_tweak_conservative/
└── weight_influence_analyzer.py         # Analysis script
```

## 🚀 How to Use

### Option 1: Easy Way (Recommended)
```bash
cd /media/hdd/usr/leyla/Unlearn-Saliency
python run_weight_analysis.py
```

### Option 2: Direct Way
```bash
cd /media/hdd/usr/leyla/Unlearn-Saliency
python weight_influence_analyzer.py
```

### Option 3: Custom Analysis
```python
from weight_influence_analyzer import WeightInfluenceAnalyzer

# Initialize analyzer
analyzer = WeightInfluenceAnalyzer(
    baseline_model_path="models/resnet50_pretrained.pth",
    experiment_models_dir="results"
)

# Analyze specific experiment
analysis = analyzer.analyze_layer_sensitivity("random_forgetting_10percent_RL")

# Or run comprehensive analysis
full_results = analyzer.generate_comprehensive_report()
```

## 📊 What You'll Get

### 1. Comprehensive JSON Report
`experiments/random_forgetting/weight_analysis/comprehensive_weight_analysis.json`

Contains detailed analysis for each experiment:
- **Layer sensitivity**: Which layers changed most
- **Distribution changes**: How weight distributions shifted
- **Top affected weights**: Individual weights that changed most

### 2. Visual Reports
`experiments/random_forgetting/visualizations/`

Bar charts showing:
- Mean relative weight changes per layer
- Maximum relative weight changes per layer
- Comparison across all experiments

### 3. Summary Report
`experiments/random_forgetting/weight_analysis/summary_report.md`

Human-readable summary with:
- Most/least affected layers per experiment
- Top changed weights
- Experiment parameter breakdown

## 🔍 Analysis Metrics

For each experiment, you'll get:

### Layer-Level Analysis
- **Mean Absolute Change**: Average weight change magnitude
- **Max Absolute Change**: Largest single weight change
- **Mean Relative Change**: Average percentage change
- **Percentage Changed**: How many weights actually changed
- **Layer Shape**: Dimensions of each layer

### Distribution Analysis
- **KL Divergence**: How much the weight distribution shifted
- **Wasserstein Distance**: Another measure of distribution change
- **Mean/Std Changes**: How the statistical properties changed

### Individual Weight Analysis
- **Top-K Most Changed**: Specific weights that changed most
- **Layer Location**: Which layer contains the most changed weights
- **Relative Change Magnitude**: How much each weight changed

## 📋 Interpretation Guide

### What to Look For:

1. **Layer Sensitivity Patterns**
   - Are final layers (classifier) most affected?
   - Do early layers (feature extractors) change much?
   - Which layers are most stable?

2. **Forgetting Ratio Effects**
   - Do 30% experiments show more change than 10%?
   - Is there a linear relationship?

3. **Method Comparisons**
   - Does RL vs GA affect different layers?
   - Which method causes more targeted changes?

4. **Conservation vs Aggressive**
   - Do conservative methods preserve more weights?
   - Where do aggressive methods make the biggest changes?

## ⚠️ Requirements

Make sure you have:
- `torch` (PyTorch)
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`

Install missing packages:
```bash
pip install torch numpy matplotlib seaborn scipy
```

## 🐛 Troubleshooting

### "Model not found" Error
- Check that your baseline model exists: `models/resnet50_pretrained.pth`
- Ensure experiment directories contain `RLcheckpoint.pth.tar` files

### "No experiments found" Error
- Make sure experiment directories have 'random_forgetting' in their names
- Check that the directories actually contain model files

### Memory Issues
- The script loads multiple models - you may need 8GB+ RAM
- Consider analyzing experiments one at a time if memory is limited

## 🎯 Research Questions This Answers

1. **Which layers are most affected by random data forgetting?**
2. **How do different forgetting ratios (10%, 20%, 30%) impact weights?**
3. **What's the difference between RL and GA unlearning methods?**
4. **Do conservative vs aggressive settings make a difference?**
5. **Are changes localized or distributed across the network?**
6. **How much do weight distributions shift during forgetting?**

Run the analysis and explore the results to answer these questions! 🚀