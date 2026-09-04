<div align="center">

# 🧠 Brain Tumor MRI Classification with Explainable Deep Learning

**Multi-class brain tumor classification from MRI, benchmarking CNN / ResNet50 / EfficientNet-B0 / ViT backbones and explaining every prediction with Grad-CAM and Saliency Maps.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>
Designed to run end-to-end in a Kaggle GPU kernel or locally (CUDA / MPS / CPU) using PyTorch and torchvision.

## Key features
- Transfer-learning baselines: ResNet50, EfficientNet‑B0
- From-scratch CNN baseline for benchmarking
- Explainability: Grad‑CAM (pytorch‑grad‑cam) and Captum saliency maps
- Config-driven training loop with checkpointing, early stopping, LR scheduling, and mixed precision (CUDA)
- Simple dataset loader for `masoudnickparvar/brain-tumor-mri-dataset` (Kaggle)

## Stack
- Language: Python 3.8+
- Framework: PyTorch (torch + torchvision)
- Notable libraries: torchvision, pytorch-grad-cam, captum, scikit-learn, pillow, matplotlib, seaborn



## What the notebook does
1. Detects dataset under `data/` (expects `Training/` and `Testing/` with class folders)
2. Builds train/val/test dataloaders and computes class weights
3. Defines models: CNN baseline, ResNet50, EfficientNet‑B0, (optional ViT)
4. Runs configurable training loop with checkpointing and early stopping
5. Evaluates on test set and produces metrics + confusion matrix
6. Produces explainability visualizations using Grad‑CAM and saliency maps

## Dataset (expected layout)
The notebook expects the Kaggle dataset `masoudnickparvar/brain-tumor-mri-dataset` with this layout:

| Class | Description |
|---|---|
| `glioma` | Tumor arising from glial cells |
| `meningioma` | Tumor arising from the meninges |
| `pituitary` | Tumor of the pituitary gland |
| `notumor` | Healthy / no visible tumor |

**Expected directory layout** (see `src/datasets/prepare_dataset.py`):

```
data/raw/
├── Training/
│   ├── glioma/       *.jpg
│   ├── meningioma/   *.jpg
│   ├── notumor/      *.jpg
│   └── pituitary/    *.jpg
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

#### Sample Data
<img width="1370" height="694" alt="image" src="https://github.com/user-attachments/assets/5268d571-084d-4229-8193-2a1110a236f4" />


## System Architecture

```mermaid
flowchart TD
    A[Raw MRI Images<br/>Training/ + Testing] --> B[Stratified Split<br/>80% / 10% / 10%]
    B --> C1[Train Set]
    B --> C2[Val Set]
    B --> C3[Test Set]

    C1 --> D[Albumentations Pipeline<br/>Resize · Normalize · Augment]
    C2 --> E[Albumentations Pipeline<br/>Resize · Normalize]
    C3 --> E

    D --> F{Model Factory}
    F --> G1[CNN Baseline<br/>4-block ConvNet]
    F --> G2[ResNet50<br/>ImageNet pretrained]
    F --> G3[EfficientNet-B0<br/>ImageNet pretrained]
    F --> G4[ViT<br/>Vision Transformer]

    G1 --> H[Trainer]
    G2 --> H
    G3 --> H
    G4 --> H
    E --> H

    H --> H1[AdamW + Cosine LR Schedule]
    H --> H2[Mixed Precision + Grad Clipping]
    H --> H3[Early Stopping on val F1]
    H --> H4[Checkpointing]
    H --> I[(W&B + TensorBoard<br/>Experiment Tracking)]

    H4 --> J[Evaluation Suite]
    J --> J1[Accuracy / Precision / Recall / F1]
    J --> J2[ROC-AUC]
    J --> J3[Confusion Matrix]
    J --> J4[Error Analysis<br/>Misclassified Samples]

    H4 --> K[Explainability Suite]
    K --> K1[Grad-CAM / Grad-CAM++]
    K --> K2[Saliency Maps]
    K1 --> L[Visualization Dashboard]
    K2 --> L

    J --> M[Benchmark Comparison Table]
    L --> N[README / Report Figures]
```

## Metrics & evaluation
#### Plot training curves
<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/df056fc3-1dd5-4afc-a775-75dbd83c091d" />

#### Confusion Matrix
<img width="1253" height="490" alt="image" src="https://github.com/user-attachments/assets/dcc5c94a-06d3-4143-a199-b0a0934cb5a5" />

#### Explainability — Grad-CAM & Saliency Maps
<img width="1016" height="2390" alt="image" src="https://github.com/user-attachments/assets/55b71a29-f32d-498f-b13a-280a3a93603b" />


## Key Ideas (Future Work)
### Federated Multimodal Explainable Medical Foundation Model (FME-MFM).
(Multimodal Foundation Models + Explainable AI + Federated Learning for Medical Image Classification)

- Build a privacy-preserving foundation model that learns jointly from medical images, radiology reports, and clinical records across multiple hospitals, while producing clinically interpretable explanations for every diagnosis.
