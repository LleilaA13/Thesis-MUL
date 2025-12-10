# Thesis-MUL: Saliency-Based Machine Unlearning with Feature Visualization

This repository contains the implementation and experiments for my thesis research on the intersection of machine unlearning and feature visualization. The work explores how Saliency-based Unlearning (SalUn) affects internal neural representations and demonstrates the critical role of feature visualization in validating knowledge removal.

## Overview

Machine unlearning has become essential for privacy compliance and responsible AI deployment. This research investigates how targeted knowledge removal manifests in the internal representations of deep neural networks, using feature visualization as both a diagnostic tool and validation mechanism for unlearning effectiveness.

## Research Focus

The thesis explores the fundamental relationship between:
- **Feature Visualization**: Understanding how networks encode visual concepts
- **Machine Unlearning**: Selectively removing specific knowledge without complete retraining
- **Representational Analysis**: Examining how unlearning affects neural activations across network layers

## Experimental Setup

### Architecture 1: InceptionV3 + ImageNet Cat Classes
- **Model**: Pre-trained InceptionV3 (via Lucent implementation)
- **Dataset**: ImageNet
- **Target Classes**: Cat breeds (classes 281-285)
  - Tabby cats
  - Persian cats  
  - Siamese cats
  - Egyptian cats
  - Cougar/mountain lions
- **Visualization Tool**: Lucent for activation maximization and layer analysis

### Architecture 2: ResNet50 + Vehicle Classification
- **Model**: ResNet50
- **Task**: Vehicle classification and forgetting
- **Focus**: Automotive category removal while preserving other vehicle classes
- **Files**: Results stored in `results/resnet50_vehicles_forgetting/`

## Methodology: SalUn (Saliency-based Unlearning)

### Core Approach
1. **Saliency Computation**: Calculate gradient magnitudes for target classes
2. **Weight Selection**: Identify most influential parameters for forget classes  
3. **Selective Modification**: Fine-tune only salient weights while freezing others
4. **Random Label Assignment**: Assign random labels to forget-set samples during retraining

### Key Innovation
Unlike traditional approaches that modify all parameters, SalUn targets only the most relevant weights, achieving:
- Efficient knowledge removal
- Preserved performance on retained classes
- Computational efficiency compared to full retraining

## Feature Visualization Analysis

### Pre/Post Unlearning Comparisons
- **Layer-wise Analysis**: Examine changes from early edge detectors to high-level semantics
- **Activation Patterns**: Compare neuron responses before and after unlearning
- **Feature Maps**: Visualize how targeted concepts are erased from internal representations

### Validation Framework
- **Traditional Metrics**: Classification accuracy on forget/retain sets
- **Visual Validation**: Feature visualization reveals residual knowledge invisible to accuracy metrics
- **Completeness Assessment**: Identify incomplete unlearning through activation analysis

## Repository Structure

```
├── results/
│   └── resnet50_vehicles_forgetting/
│       ├── 0checkpoint.pth.tar          # Model checkpoints
│       └── 0model_SA_best.pth.tar       # Best saliency-aware model
├── notebooks/                           # Jupyter notebooks for experiments
├── src/                                # Source code implementation
└── visualizations/                     # Generated feature visualizations
```

## Key Findings

### 1. Visualization as Validation
Feature visualization provides superior validation for unlearning completeness compared to accuracy metrics alone, revealing subtle knowledge retention patterns.

### 2. Multi-Layer Impact
Effective unlearning requires coordinated changes across network layers - incomplete removal often shows residual activations in deeper layers.

### 3. Architecture Generalizability
SalUn demonstrates effectiveness across different architectures (InceptionV3, ResNet50) and domains (ImageNet classification, vehicle recognition).

### 4. Representational Understanding
The research establishes that understanding internal feature dynamics is crucial for developing trustworthy unlearning methods.

## Technologies Used

- **Deep Learning**: PyTorch
- **Visualization**: Lucent (modified InceptionV3 implementation)
- **Unlearning**: SalUn implementation
- **Analysis**: Jupyter notebooks for experimental workflows
- **Model Storage**: Git LFS for large model files (*.pth.tar)


## Thesis Contributions

1. **Novel Framework**: Links representational analysis with privacy-preserving machine learning
2. **Validation Methodology**: Establishes feature visualization as essential for unlearning verification
3. **Multi-Architecture Analysis**: Demonstrates generalizability across network designs
4. **Practical Applications**: Provides tools for trustworthy knowledge removal in production systems

## Future Work

- Extension to other architectural families (Transformers, Vision Transformers)
- Real-world deployment scenarios and privacy guarantees
- Automated detection of incomplete unlearning through visualization analysis
- Integration with differential privacy frameworks

## License

MIT License - See LICENSE file for details

---

*This repository supports the thesis research on "Saliency-Based Machine Unlearning with Feature Visualization Analysis" demonstrating how internal neural representations change during targeted knowledge removal.*
This repository contains the official code and experimental results my thesis porject. The project investigates the internal changes in a ResNet-50 model trained on TinyImageNet when subjected to data forgetting, specifically through random data removal, using SalUn.

The primary goal is to move beyond surface-level accuracy metrics and analyze the mechanistic impact of unlearning on the model's weights and learned features. We use saliency-based unlearning techniques and visualize the effects using feature visualization tools like Lucent.


   Machine Unlearning Implementation: Implements a Random Labels (RL) unlearning strategy to force a model to "forget" a subset of its training data.

   Targeted Data Forgetting: Scripts to forget a random 10%, 20%, or 30% of the TinyImageNet dataset.

   Weight Influence Analysis: The framework is designed to compare the model's weights before and after unlearning to identify which layers and channels are most affected.

   Feature Visualization: Integrates with the Lucent library to provide a qualitative understanding of how a neuron's "preferred" visual patterns change after unlearning.

   Comprehensive Experiment Suite: Includes scripts for running various unlearning configurations (e.g., conservative, aggressive) and evaluating their impact on retain and forget set accuracy.
