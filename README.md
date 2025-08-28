# Thesis: Saliency-Based Unlearning on InceptionV1

This repository contains the code and experiments from my thesis work on applying saliency-based unlearning (SalUn) to a convolutional neural network trained on ImageNet. The goal is to remove all knowledge of certain classes — in this case, cat classes — from the model, and study how this affects its internal representations.

---

## Overview

The main idea is to use the [SalUn method](https://github.com/OPTML-Group/Unlearn-Saliency) to erase a subset of classes from a trained classifier without retraining from scratch. After unlearning, I use [Lucent](https://github.com/greentfrapp/lucent) to visualize changes in the model's layers and neurons.

I'm using a modified version of Lucent’s `inceptionv3`, pre-trained on full ImageNet. The classes I target for forgetting are ImageNet classes 281–285 (various cat breeds).

