This repository contains the official code and experimental results my thesis porject. The project investigates the internal changes in a ResNet-50 model trained on TinyImageNet when subjected to data forgetting, specifically through random data removal, using SalUn.

The primary goal is to move beyond surface-level accuracy metrics and analyze the mechanistic impact of unlearning on the model's weights and learned features. We use saliency-based unlearning techniques and visualize the effects using feature visualization tools like Lucent.


   Machine Unlearning Implementation: Implements a Random Labels (RL) unlearning strategy to force a model to "forget" a subset of its training data.

   Targeted Data Forgetting: Scripts to forget a random 10%, 20%, or 30% of the TinyImageNet dataset.

   Weight Influence Analysis: The framework is designed to compare the model's weights before and after unlearning to identify which layers and channels are most affected.

   Feature Visualization: Integrates with the Lucent library to provide a qualitative understanding of how a neuron's "preferred" visual patterns change after unlearning.

   Comprehensive Experiment Suite: Includes scripts for running various unlearning configurations (e.g., conservative, aggressive) and evaluating their impact on retain and forget set accuracy.
