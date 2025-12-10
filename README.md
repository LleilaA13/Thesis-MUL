# Thesis-MUL: Saliency-Based Machine Unlearning

This repository implements and evaluates saliency-based machine unlearning (SalUn) with a focus on two experiment families:
- **Class-wise forgetting**: remove specific semantic classes (cats, dogs, vehicles).
- **Random-data forgetting**: remove random subsets (10/20/30%) of Tiny ImageNet.

The codebase pairs quantitative evaluation with feature visualization (Lucent) to verify whether knowledge was actually erased inside the network.

## Quick Start

```bash
conda env create -f environment.yml
conda activate salun

# Typical data layout
datasets/tiny-imagenet-200/

# Run a random-data forgetting notebook
jupyter notebook notebooks/random_data/20_random_data.ipynb
```

## Repository Map (organized by forgetting mode)

```
experiments/
├── class_wise/
│   ├── masks/            # Saliency masks per class family
│   ├── models/           # Class-wise checkpoints
│   ├── indices/          # Forget-set indices & masks
│   ├── scripts/          # Training/unlearning entrypoints
│   └── notebooks/        # (reserved) class-wise notebooks
├── random_data/
│   ├── masks/            # Masks for random 10/20/30% forgetting
│   ├── results/          # Quantitative + good_results
│   ├── forgotten_images/ # Exported forgotten samples
│   ├── scripts/          # Pipelines for random forgetting
│   └── notebooks/        # (reserved) random-data notebooks
└── common/
   └── resnet50_pretrained.pth  # Base model

notebooks/
├── random_data/          # 10/20/30_random_data, comparison_experiments, config.py
├── class_wise/           # Resnet18/, resnet-50/, inceptionv3/, legacy/
└── common/               # diversity, feature_inversion, modelzoo, neuron_interaction, tutorial

src/                      # Training, data, utils
scripts/                  # Analysis helpers (forgotten data, visualization)
labels/, datasets/        # Tiny ImageNet labels/data
models/, results/, visuals, plots  # Outputs and figures
```

## Key Entry Points

- **Random-data forgetting notebooks**: `notebooks/random_data/10_random_data.ipynb`, `20_random_data.ipynb`, `30_random_data.ipynb`, `comparison_experiments.ipynb`
- **Class-wise notebooks**: `notebooks/class_wise/resnet-50/…`, `notebooks/class_wise/Resnet18/…`, `notebooks/class_wise/inceptionv3/…` (legacy variants kept in `notebooks/class_wise/legacy/`)
- **Scripts (class-wise)**: `experiments/class_wise/scripts/resnet50_unlearn_*.py`, `generate_all_masks.py`, `generate_mask_class.py`, `main_random_class.py`
- **Scripts (random-data)**: see `experiments/random_data/scripts/` (add new runs here if needed)
- **Base model**: `experiments/common/resnet50_pretrained.pth`

## Data & Models

- Dataset: Tiny ImageNet under `datasets/tiny-imagenet-200/`
- Forget-set metadata: `experiments/class_wise/indices/*.pt`
- Masks: `experiments/class_wise/masks/` and `experiments/random_data/masks/`
- Checkpoints: `experiments/class_wise/models/` plus base model in `experiments/common/`
- Results: `experiments/random_data/results/good_results/` contains RL and conservative variants

## Workflow

1) Choose forgetting mode
  - Class-wise: run scripts in `experiments/class_wise/scripts/`
  - Random-data: run notebooks in `notebooks/random_data/`

2) Train/unlearn
  - Use SalUn with random-label (RL) variants (see notebook configs)

3) Validate
  - Quantitative: accuracy on retain vs forget splits (stored in results)
  - Qualitative: Lucent activation grids and neuron visualizations (common notebooks)

## Notes

- Large artifacts are ignored by `.gitignore`; keep big checkpoints under `experiments/` or `models/`.
- `CLEANUP_GUIDE.md` and `CLEANUP_SUMMARY.md` describe how the repo was reorganized.
- Legacy class-wise notebooks are preserved under `notebooks/class_wise/legacy/` for reference.

## License

MIT License. See `LICENSE`.
