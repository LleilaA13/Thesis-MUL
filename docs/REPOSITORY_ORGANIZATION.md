# Repository Organization Guide

## 📁 Directory Structure Overview

The repository has been reorganized for clarity, maintainability, and logical separation of concerns:

```
Unlearn-Saliency/
├── 📁 core/                    # Core unlearning implementations
├── 📁 analysis/                # Analysis tools and visualizations  
├── 📁 experiments/             # Experimental results and models
├── 📁 datasets/                # Data and labels
├── 📁 docs/                    # Documentation and guides
├── 📁 scripts/                 # Utility and pipeline scripts
├── 📄 README.md                # Main project README
├── 📄 requirements_research.txt # Python dependencies
└── 📄 LICENSE                  # Project license
```

---

## 📂 Detailed Directory Contents

### 🔧 `core/` - Core Implementations
Contains the main unlearning algorithm implementations:
```
core/
└── Classification/             # Main classification-based unlearning
    ├── main_train.py          # Training script
    ├── main_forget.py         # Forgetting/unlearning script
    ├── main_random.py         # Random forgetting experiments
    ├── models/                # Model architectures
    ├── trainer/               # Training utilities
    ├── unlearn/               # Unlearning algorithms
    ├── evaluation/            # Evaluation metrics
    └── pruner/                # Pruning-based methods
```

### 🔬 `analysis/` - Analysis Tools
All analysis, visualization, and research tools:
```
analysis/
├── tools/                     # Analysis utilities
│   ├── weight_influence_analyzer.py     # Core weight analysis
│   └── channel_level_analyzer.py       # Channel-level analysis
├── visualizations/            # Visualization scripts
│   ├── comprehensive_weight_visualizer.py
│   ├── improved_layer_visualizer.py
│   └── enhanced_visualization.py
└── weight_analysis/          # Weight analysis workflows
    ├── analyze_good_results.py
    ├── run_weight_analysis.py
    └── run_complete_analysis.py
```

### 🧪 `experiments/` - Experimental Results
All experimental data, models, and results:
```
experiments/
├── results/                  # Experimental results and outputs
├── models/                   # Trained and unlearned models
│   ├── resnet18/            # ResNet18 experiments
│   ├── inceptionv3_cat_forgetting/
│   ├── resnet50_cats_forgetting/
│   └── resnet50_vehicles_CORRECT/
├── masks/                    # Pruning masks and forgetting masks
├── good_results_weight_analysis/  # Weight analysis results
├── channel_analysis/         # Channel-level analysis results
└── unlearn.log              # Experiment logs
```

### 💾 `datasets/` - Data and Labels
Dataset files and preprocessing:
```
datasets/
├── tiny-imagenet-200/       # Tiny ImageNet dataset
├── cifar-10-python.tar.gz  # CIFAR-10 dataset
├── train_ys.pth            # Training labels
└── val_ys.pth              # Validation labels
```

### 📖 `docs/` - Documentation
Documentation, guides, and visual assets:
```
docs/
├── README_Research.md       # Research documentation
├── WEIGHT_ANALYSIS_GUIDE.md # Weight analysis guide
└── Images/                  # Figures and visual assets
    ├── church.jpg
    ├── teaser-v2.png
    └── transition_new.gif
```

### ⚙️ `scripts/` - Utility Scripts
Utility scripts and pipeline automation:
```
scripts/
├── research_pipeline.py     # Main research pipeline
├── run_research_pipeline.sh # Shell script runner
├── demo_channel_lucent.py   # Lucent visualization demo
└── lucent_weight_integration.py # Lucent integration
```

---

## 🚀 Quick Start Guide

### 1. **Training Models**
```bash
cd core/Classification
python main_train.py --dataset tiny-imagenet --model resnet50
```

### 2. **Running Unlearning**
```bash
cd core/Classification
python main_forget.py --model resnet50 --forget-ratio 0.1
```

### 3. **Weight Analysis**
```bash
cd analysis/weight_analysis
python analyze_good_results.py
```

### 4. **Generating Visualizations**
```bash
cd analysis/visualizations
python improved_layer_visualizer.py
```

### 5. **Complete Research Pipeline**
```bash
bash scripts/run_research_pipeline.sh
```

---

## 🔍 Key Features of New Organization

### ✅ **Benefits**
- **Clear separation of concerns**: Core algorithms vs analysis vs experiments
- **Logical grouping**: Related files are together
- **Scalability**: Easy to add new experiments or analysis tools
- **Navigation**: Intuitive directory names and structure
- **Maintenance**: Easier to find and update specific components

### 🗂️ **File Categories**
- **Core Code**: `core/Classification/` - The main unlearning implementation
- **Analysis Tools**: `analysis/tools/` - Reusable analysis utilities
- **Visualizations**: `analysis/visualizations/` - Plotting and visualization
- **Results**: `experiments/` - All experimental outputs and models
- **Documentation**: `docs/` - Guides, READMEs, and visual assets
- **Utilities**: `scripts/` - Automation and utility scripts

### 🧹 **Cleanup Performed**
- ❌ Removed unused DDPM and SD directories
- ❌ Deleted duplicate and empty directories
- ❌ Cleaned up `__pycache__` and `.pyc` files
- ❌ Consolidated scattered experiment files
- ✅ Grouped related functionality
- ✅ Created logical hierarchy
- ✅ Maintained all important research assets

---

## 📋 Migration Notes

### **Moved Files**
- `*_analyzer.py` → `analysis/tools/`
- `*_visualizer.py` → `analysis/visualizations/`
- `Classification/` → `core/Classification/`
- `experiments/`, `results/`, `masks/`, `models/` → `experiments/`
- `README_Research.md`, `Images/` → `docs/`
- Pipeline scripts → `scripts/`

### **Removed**
- `DDPM/` and `SD/` directories (unused)
- Duplicate `src/` directory
- Empty directories and cache files
- Scattered temporary files

### **Path Updates Needed**
If you have scripts with hardcoded paths, update them to use the new structure:
- Old: `./analyze_good_results.py` → New: `./analysis/weight_analysis/analyze_good_results.py`
- Old: `./Classification/main_train.py` → New: `./core/Classification/main_train.py`
- Old: `./experiments/` → New: `./experiments/` (structure changed internally)

---

## 🎯 Next Steps

1. **Update import statements** in scripts to match new paths
2. **Test key workflows** to ensure everything still works
3. **Update documentation** to reflect new structure
4. **Consider adding `__init__.py` files** for Python package structure
5. **Update CI/CD scripts** if any exist

This organization makes the repository much more maintainable and easier to navigate for research purposes!