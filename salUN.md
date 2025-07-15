# SALUN: Empowering Machine Unlearning via Gradient-Based Weight Saliency

*ICLR 2024 — Chongyu Fan et al.*

## 🔍 Problem Motivation

Machine Unlearning (MU) aims to **remove the influence of specific data** from a trained model, aligning with privacy laws like GDPR’s “right to be forgotten.” 

Existing MU methods fall into two main categories:

- **Exact Unlearning** (Retraining from scratch): Accurate, but computationally expensive.
- **Approximate Unlearning**: More efficient, but often unstable or ineffective.

Most prior work focuses on **image classification** tasks. However, with the rise of **generative models** like Stable Diffusion, there's a growing need for effective MU in **image generation** as well (e.g., to remove harmful concepts like nudity).

---

## 💡 Core Idea: Weight Saliency

Inspired by **input saliency maps** (used in model interpretability), SALUN introduces the idea of **weight saliency**:

> Focus MU on only the **most influential model weights** instead of modifying the entire model.

### How it works:

1. Compute the **gradient of the forgetting loss** with respect to the model weights.
2. Use a **hard threshold** to identify which weights are “salient.”
3. Apply MU techniques (like random labeling) only to those salient weights.

---

## 🧠 Methodology

### Weight Saliency Map

Let \( D_f \) be the dataset to be forgotten, and \( \ell_f(\theta; D_f) \) the forgetting loss.

The saliency mask is defined as:

\[
m_s = 1\left( \left| \nabla_\theta \ell_f(\theta; D_f) \right| \geq \gamma \right)
\]

Where:
- \( m_s \) is a binary mask selecting salient weights.
- \( \gamma \) is a threshold (e.g., median of gradient magnitudes).

### Model Update Rule

\[
\theta_u = m_s \cdot (\Delta\theta + \theta_o) + (1 - m_s) \cdot \theta_o
\]

This ensures only the salient weights are updated during unlearning.

### Integration with Random Labeling (RL)

- **For Classification**: Apply RL on \( D_f \) with incorrect labels.
- **For Generation**: Associate harmful prompts (e.g., "nudity") with unrelated images.

---

## 🧪 Experimental Results

### Image Classification (e.g., CIFAR-10)

- SALUN consistently achieves the **smallest performance gap** compared to retraining (exact unlearning).
- Strong balance across:
  - **Unlearning Accuracy (UA)**
  - **Remaining Accuracy (RA)**
  - **Testing Accuracy (TA)**
  - **Membership Inference Attack (MIA) Resistance**
  - **Runtime Efficiency (RTE)**

### Image Generation (e.g., Stable Diffusion)

- Nearly **100% unlearning accuracy** on harmful concepts (e.g., nudity).
- Outperforms baselines like **ESD** and **FMN** while preserving generation quality (low FID scores).
- Effective in **concept-wise forgetting** — e.g., removing NSFW concepts from prompts.

---

## 🎯 Why It Matters

- **First unified MU framework** effective for both classification *and* generation.
- Improves **efficiency, stability, and adaptability** of MU.
- Critical for **safe and privacy-preserving AI**, especially in open-ended generation tasks.

---

## 📝 Use in Thesis

Consider structuring your thesis section on this work as:

1. Introduction to Machine Unlearning
2. Limitations of Prior Work
3. The Innovation of Weight Saliency
4. Technical Approach & Key Equations
5. Empirical Results & Comparisons
6. Applications in Generative Models
7. Broader Implications and Future Work
