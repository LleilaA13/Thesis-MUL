# Methodology

This document describes the methodology used in the experiments for random data forgetting and subsequent analysis. It synthesizes the procedures implemented in the repository, including dataset handling, forgetting protocol, unlearning method (RL), analysis pipeline (RealInfluenceAnalyzer), and Lucent-based feature visualization.

## 1. Dataset
- Dataset: Tiny ImageNet (200 classes)
- Train/Validation split: Provided by Tiny ImageNet standard structure under `datasets/tiny-imagenet-200/`.
- Data loading: Standard PyTorch `ImageFolder`/custom dataset utilities are used for loading and batching.

## 2. Forgetting Protocol
- Forgetting type: Random data forgetting (10% of training samples by default in the experiments folder).
- Selection method: Samples selected for forgetting are chosen randomly using a seeded RNG. The selection is represented as a mask/indices file (e.g., `masks/random_forgetting_10percent/with_0.6.pt`) passed to the unlearning script via `--mask_path`.
- Replacement strategy: During unlearning the labels of forgotten samples are randomized (or otherwise altered) as per the `RL` unlearning implementation.
- Parameters used in experiments (example command):

```
python main_random.py --unlearn RL --dataset TinyImagenet --arch resnet50 \
  --data_dir ../datasets/tiny-imagenet-200 --unlearn_epochs 15 --unlearn_lr 0.005 \
  --num_indexes_to_replace 10000 --model_path ../models/resnet50_pretrained.pth \
  --save_dir ../results/random_forgetting_10percent_RL_tweak_conservative \
  --mask_path ../masks/random_forgetting_10percent/with_0.6.pt
```

## 3. Unlearning Method
- Algorithm: RL (Re-Labeling / Randomization-based unlearning) implemented in `core/Classification/unlearn/RL.py`.
- Process: For the specified forget set, targets are randomized to new labels sampled uniformly from the class set. The model is then trained (or fine-tuned) with a modified training loader that mixes preserved and randomized data.
- Mask usage: The `--mask_path` loads a parameter mask in some experiments, but data selection indices should be saved explicitly (the repository contains scripts to export forgotten indices during masking experiments).

## 4. Analysis Pipeline
- Analyzer: `RealInfluenceAnalyzer` (analysis/tools/real_influence_analyzer.py) compares baseline and unlearned models.
- Granularity:
  - Layer-level: mean absolute and relative weight changes per layer.
  - Channel-level: aggregated per-channel influence measures.
  - Weight-level: top-K individual weight changes.
- Outputs:
  - Per-experiment JSON files with top layers/channels/weights and lucent targets.
  - A multi-experiment comparison JSON summarizing severity ranking and common affected layers.

## 5. Lucent Feature Visualization
- Library: Lucent (feature visualization toolkit based on TensorFlow and PyTorch adapters), installed from GitHub.
- Targets: `lucent_targets_real.json` contains the layer/channel objectives generated from the analyzer results.
- Visualization modes:
  - Channel maximization: `objectives.channel(layer, channel)`
  - Direction objective (scar visualization): `objectives.direction(layer, vector)` computed from activation differences.
- Notebook: `notebooks/10_random_data.ipynb` demonstrates generating proxy images, computing activation deltas, and visualizing "scars".

## 6. Evaluation Metrics
- Accuracy on retained and forgotten subsets before and after unlearning.
- Percentage of weights changed, maximum and average changes.
- Visual inspection of Lucent outputs for qualitative assessment.

## 7. Reproducibility
- Environment: Use the provided `environment.yml` or Conda environment `salUN` to replicate dependencies (PyTorch, torchvision, Lucent, pandas, matplotlib).
- Configuration: Use `notebooks/config.py` for notebook paths to make notebooks portable between local and Colab.
- Saving indices: Ensure the script saves `forgotten_indices.pt` alongside experiment results for later extraction of forgotten images.

## 8. Practical Notes
- BatchNorm running statistics are highly sensitive; treat them separately when evaluating unlearning effects.
- Lucent visualizations may produce non-deterministic images; fix seeds and document transforms for reproducibility.

---

This methodology file is a living document: refine it with exact hyperparameters, random seeds, and citations as you finalize the thesis.