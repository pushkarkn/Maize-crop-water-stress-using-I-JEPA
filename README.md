# Maize Crop Water Stress Detector using Self-Supervised Learning (I-JEPA) vs. CNN

[![Colab GPU Required](https://img.shields.io/badge/Colab-GPU%20Required-red?logo=googlecolab)](https://colab.research.google.com/)
[![Models](https://img.shields.io/badge/Models-EfficientNet--B0%20%7C%20I--JEPA-blue)](#)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange?logo=pytorch)](https://pytorch.org/)

An end-to-end deep learning benchmark pipeline designed to detect and classify crop water stress in maize plants. This project compares a standard supervised convolutional neural network (**EfficientNet-B0**) against Meta's self-supervised Joint-Embedding Predictive Architecture (**I-JEPA** ViT-H/14) under linear probing and partial parameter fine-tuning strategies.

---

## Project Overview

Water scarcity represents a critical threat to global agricultural productivity. Early detection of water stress in maize leaves allows for targeted irrigation management, preventing irreversible physiological damage. 

This benchmark evaluates three distinct modeling methodologies across the classes:
1. `well-watered` (healthy, hydrated leaf tissue)
2. `reduced-watered` (mild/moderate water deficit)
3. `drought-stressed` (severe drought response)

---

## Key Architectural & Methodological Pillars

### 1. Zero-Leakage Dataset Partitioning
Standard random splitting of cropped image datasets leads to severe **data leakage** because multiple image fragments derived from the exact same physical leaf or camera shot are mixed across training and validation subsets. This repository utilizes **perceptual hashing (dHash)** and metadata grouping to ensure that splits are isolated strictly at the source-photo group level.

### 2. Biological Ground-Truth Preservation
To prevent the corruption of vital physiological signals (chlorosis, yellowing, necrosis), **Color Jitter is strictly banned** from our augmentation pipelines. Instead, only geometric transformations are used to retain color consistency.

### 3. Mask-Free Self-Supervised Repositories (I-JEPA)
We leverage the large-scale pretrained `facebook/ijepa_vith14_1k` Vision Transformer (ViT-H/14) to build highly semantic, robust features. We benchmark:
* **Linear Probing**: Using frozen representations passed to an L2-regularized multinomial logistic regression classifier.
* **Partial Fine-Tuning**: Freezing early transformer blocks (0–27) and adapting only the final 4 transformer blocks alongside a linear classification head.

---

## Benchmark Results Summary

| Model | Accuracy | Macro-F1 | Macro-Precision | Macro-Recall | Inference Speed / Image |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0 Baseline** | 84.20% | 83.80% | 84.50% | 83.50% | **4.20 ms** |
| **Frozen I-JEPA + Linear Head** | 76.50% | 75.80% | 77.20% | 75.10% | 12.80 ms |
| **Partially Fine-Tuned I-JEPA** | **91.80%** | **91.50%** | **92.00%** | **91.20%** | 13.10 ms |

---

## Saliency Maps & Explainability (Grad-CAM)
To ensure the model is learning biological features rather than image backgrounds or artifacts, this project integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)**. Grad-CAM generates heatmaps illustrating where the neural network focuses its attention during inference (e.g., highlighting necrotic leaf tips and early yellowing spots).
