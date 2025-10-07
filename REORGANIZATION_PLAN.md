# 📁 Manual Repository Reorganization Plan

## Current Structure Analysis
Your repository has these main issues:
- Scripts scattered in root directory
- Mixed experimental files with core code
- No clear separation between different types of content
- Missing comprehensive documentation

## 🎯 Recommended New Structure

```
Unlearn-Saliency/
├── README.md                          # Main project documentation
├── SETUP.md                          # Installation and setup guide
├── environment.yml                   # Conda environment
├── requirements.txt                  # Pip requirements
│
├── 📁 src/                           # Core source code
│   ├── __init__.py
│   ├── classification/               # Your existing core framework
│   │   ├── models/
│   │   ├── unlearn/
│   │   ├── evaluation/
│   │   └── ...
│   └── utils/                        # Shared utilities
│       ├── __init__.py
│       ├── data_utils.py
│       ├── model_utils.py
│       └── plot_utils.py
│
├── 📁 experiments/                   # Experiment scripts
│   ├── __init__.py
│   ├── resnet50_unlearn.py          # Move from root
│   ├── inceptionv3_unlearn.py       # Move from root
│   ├── create_vehicle_forget_mask.py
│   └── configs/                     # Configuration files
│       ├── resnet50_config.yaml
│       └── inception_config.yaml
│
├── 📁 notebooks/                    # Analysis notebooks
│   ├── exploratory/                # Data exploration
│   ├── visualization/               # Result visualization
│   ├── Resnet18/                    # Your existing notebooks
│   └── analysis/                    # Thesis analysis
│
├── 📁 scripts/                      # Utility scripts
│   ├── __init__.py
│   ├── download_data.sh
│   ├── setup_environment.sh
│   ├── run_experiments.sh
│   ├── generate_plots.py            # Your plotting scripts
│   └── evaluation/
│       ├── run_mia_evaluation.py
│       └── generate_metrics.py
│
├── 📁 data/                         # Data directory
│   ├── raw/                         # Original datasets
│   │   └── tiny-imagenet-200/       # Move from datasets/
│   ├── processed/                   # Processed data
│   └── masks/                       # Generated masks
│       ├── resnet50_vehicles_forgetting/
│       └── inceptionv3_cat_forgetting/
│
├── 📁 models/                       # Trained models
│   ├── pretrained/                  # Downloaded pretrained models
│   ├── checkpoints/                 # Training checkpoints
│   └── final/                       # Final trained models
│
├── 📁 results/                      # Experimental results
│   ├── metrics/                     # Performance metrics
│   ├── logs/                        # Training logs
│   └── evaluation/                  # Evaluation results
│
├── 📁 figures/                      # Generated figures
│   ├── thesis/                      # Figures for thesis
│   ├── presentations/               # Presentation figures
│   └── analysis/                    # Analysis plots
│
├── 📁 docs/                         # Documentation
│   ├── thesis/                      # Thesis-related docs
│   │   ├── transfer_learning_analysis.md
│   │   └── thesis_visualization_guide.md
│   ├── api/                         # Code documentation
│   └── tutorials/                   # How-to guides
│
└── 📁 tests/                        # Test files
    ├── __init__.py
    ├── test_dataset.py              # Move test_tinyimagenet.py here
    ├── test_models.py
    └── test_unlearning.py
```

## 🔄 Manual Reorganization Steps

### Step 1: Create New Directory Structure
```bash
# Create main directories
mkdir -p experiments/configs
mkdir -p scripts/evaluation
mkdir -p data/{raw,processed,masks}
mkdir -p models/{pretrained,checkpoints,final}
mkdir -p results/{metrics,logs,evaluation}
mkdir -p figures/{thesis,presentations,analysis}
mkdir -p docs/{thesis,api,tutorials}
mkdir -p tests
mkdir -p src/utils
```

### Step 2: Move Root Scripts to Appropriate Locations
```bash
# Move experiment scripts
mv resnet50_unlearn.py experiments/
mv inceptionv3_unlearn.py experiments/
mv create_vehicle_forget_mask.py experiments/

# Move test files
mv test_tinyimagenet.py tests/test_dataset.py
mv debug_tinyimagenet.py tests/

# Move plotting scripts
mv generate_thesis_plots.py scripts/
mv generate_feature_plots.py scripts/
mv generate_evaluation_plots.py scripts/
mv run_all_plots.py scripts/

# Move data
mv datasets/ data/raw/
mv masks/ data/masks/

# Move documentation
mv transfer_learning_analysis.md docs/thesis/
mv thesis_visualization_guide.md docs/thesis/
```

### Step 3: Clean Up Root Directory
```bash
# Remove scattered files (after backing up important ones)
rm -f cat_forget_indices.pt           # Move to data/processed/
rm -f vehicles_forget_indices.pt      # Move to data/processed/
rm -f marked_labels.pt                # Move to data/processed/

# Move TODO files to docs
mv TODO.md docs/
mv to_do.md docs/
mv todo2.md docs/
mv salUN.md docs/
```

### Step 4: Update Import Statements
After moving files, you'll need to update import statements:

**In experiments/resnet50_unlearn.py:**
```python
# Change this:
sys.path.append(os.path.join(current_dir, "src", "classification"))

# To this:
sys.path.append(os.path.join(current_dir, "..", "src", "classification"))
```

**In tests/test_dataset.py:**
```python
# Change this:
sys.path.append(os.path.join(current_dir, "src"))

# To this:
sys.path.append(os.path.join(current_dir, "..", "src"))
```

### Step 5: Create Configuration Files

**experiments/configs/resnet50_config.yaml:**
```yaml
model:
  arch: "resnet50"
  pretrained: true
  num_classes: 200

training:
  lr: 0.001
  epochs: 100
  batch_size: 64
  
data:
  dataset: "TinyImagenet"
  data_dir: "../data/raw/tiny-imagenet-200"
  
unlearning:
  forget_classes: ["vehicle"]
  mask_sparsity: 0.5
```

### Step 6: Create Utility Scripts

**scripts/setup_environment.sh:**
```bash
#!/bin/bash
echo "Setting up environment for Unlearn-Saliency..."
conda env create -f environment.yml
conda activate salUN
pip install -r requirements.txt
echo "Environment setup complete!"
```

**scripts/run_experiments.sh:**
```bash
#!/bin/bash
echo "Running ResNet-50 unlearning experiment..."
cd experiments
python resnet50_unlearn.py
echo "Experiment complete!"
```

### Step 7: Update README.md
Replace your current README with a comprehensive one that explains:
- Project overview
- Installation instructions
- Usage examples
- Repository structure
- Results and findings

## 🎯 Benefits of This Organization:

1. **Clear Separation**: Code vs experiments vs documentation
2. **Scalability**: Easy to add new experiments
3. **Professional**: Thesis-ready structure
4. **Maintainable**: Logical grouping of related files
5. **Reproducible**: Clear scripts and configurations
6. **Collaborative**: Easy for others to understand and contribute

## 📋 Checklist for Manual Reorganization:

- [ ] Create directory structure
- [ ] Move files to appropriate locations
- [ ] Update import statements in moved files
- [ ] Create configuration files
- [ ] Update README.md
- [ ] Test that experiments still run
- [ ] Update any hardcoded paths
- [ ] Commit changes to git

Would you like me to help you with any specific step or create template files for the new structure?