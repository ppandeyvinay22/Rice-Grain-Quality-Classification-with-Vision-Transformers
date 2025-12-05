# Rice Grain Quality Classification with Vision Transformers

> Vision Transformer (MobileViT-Small) implementation for automated rice grain quality inspection achieving **96.4% validation accuracy** on multi-class classification. PyTorch-based pipeline for FCI grain assessment system.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)

## 📋 Project Overview

This project implements a Vision Transformer model (MobileViT-Small) for automated rice grain quality classification for the Food Corporation of India (FCI). The system processes segmented rice grain images and classifies them into multiple quality categories including Idly Rice, Damaged Grains, and other rice varieties.

**Production Pipeline:**
```
U-Net Segmentation → Extract Individual Grains → MobileViT Classifier
                                                  ↓
                                      Classify ~2000 grains/batch (30-40 sec)
```

### Key Features

- ✅ **Vision Transformer Architecture**: MobileViT-Small (5.6M parameters) - lightweight and efficient
- ✅ **High Accuracy**: 96.4% validation accuracy on 12+ rice grain classes
- ✅ **Fast Inference**: Optimized for Intel NUC production deployment
- ✅ **Robust Training**: Class-weighted focal loss for handling class imbalance
- ✅ **Production Ready**: PyTorch → ONNX → OpenVINO conversion pipeline

---

## 🎯 Performance Metrics

| Metric | Value |
|--------|-------|
| **Architecture** | MobileViT-Small |
| **Parameters** | 5.6M |
| **Input Size** | 224×224×3 RGB |
| **Classes** | 12-15 rice grain categories |
| **Validation Accuracy** | **96.4%** |
| **Training Loss** | 0.0013 |
| **Validation Loss** | 0.1091 |
| **Training Time** | ~24 epochs (with early stopping) |

---

## 🏗️ Architecture Details

**MobileViT-Small** is a hybrid CNN + Transformer architecture:

```
Input (224×224×3)
    ↓
Convolutional Stem (CNN feature extraction)
    ↓
MobileViT Blocks (CNN + Self-Attention)
    ├─ Local features via 3×3 convolutions
    ├─ Patch unfolding
    ├─ Multi-head self-attention
    └─ Patch folding back to spatial
    ↓
Global Average Pooling (no CLS token)
    ↓
Classification Head → [num_classes]
```

**Why MobileViT?**
- Combines CNN efficiency with Transformer global context
- No CLS token needed (uses Global Average Pooling)
- Optimized for mobile/edge deployment
- Better accuracy-speed tradeoff than pure CNNs

---

## 📊 Dataset

- **Total Images**: ~24,000 (after balancing)
- **Image Size**: 224×224 pixels
- **Format**: RGB images of individual rice grains
- **Classes**: 12-15 rice quality categories
- **Split**: 80% Train / 10% Validation / 10% Test

### Class Distribution (Balanced)
All classes limited to ~2000 images each for balanced training.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ppandeyvinay22/Rice-Grain-Quality-Classification-with-Vision-Transformers
cd MobileVitClassification
```

2. **Create virtual environment**
```bash
virtualenv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🎓 Training

### Quick Start

**Basic training:**
```bash
python train.py
```

**Background training with nohup:**
```bash
nohup python train.py > training.log 2>&1 &
```

**Monitor training:**
```bash
tail -f training.log
```

### Training Configuration

Key hyperparameters (editable in `train.py` or config file):

```python
# Model
MODEL_NAME = "apple/mobilevit-small"
IMAGE_SIZE = 224
NUM_CLASSES = 12  # Adjust based on your dataset

# Training
BATCH_SIZE = 32
NUM_EPOCHS = 80
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

# Optimization
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 8
```

### Optimal Configuration (Recommended)

Based on extensive experimentation, these settings achieved **96.4% validation accuracy**:

| Hyperparameter | Value | Reasoning |
|----------------|-------|-----------|
| **Learning Rate** | 2e-4 | Standard for transformer fine-tuning |
| **Weight Decay** | 1e-4 | L2 regularization to prevent overfitting |
| **Batch Size** | 32 | Balance between speed and memory |
| **Optimizer** | AdamW | Adam with decoupled weight decay |
| **Warmup** | 10% of steps | Stabilizes early training |
| **Scheduler** | Cosine with warmup | Smooth LR decay |
| **Early Stopping** | Patience 8 | Stop if no improvement for 8 epochs |

### Data Augmentation

Runtime augmentations applied during training:
- Random Horizontal Flip (p=0.5)
- Random Vertical Flip (p=0.5)
- Random Rotation (±25°)
- Color Jitter (brightness ±15%)
- Random Affine (translation ±10%)

---

## 🔬 Loss Function

### Custom Focal Loss with Class Weights

The training uses a **combination of class weights and focal loss** to handle class imbalance:

**Focal Loss Formula:**
```
FL(pt) = -αt * (1 - pt)^γ * log(pt)
```

Where:
- `pt` = predicted probability for true class
- `α` = class weight (computed from inverse class frequency)
- `γ` = focusing parameter (default: 2.0)

**Why this combination?**
1. **Class Weights**: Handle class imbalance (e.g., fewer Damaged Grains samples)
2. **Focal Loss**: Focus training on hard-to-classify examples
3. **Better convergence**: Prevents majority class dominance

**Implementation:**
```python
# Class weights computed automatically from training data
class_weights = compute_class_weight('balanced', 
                                      classes=np.unique(train_labels),
                                      y=train_labels)

# Focal loss with class weights
loss = focal_loss(outputs, labels, alpha=class_weights, gamma=2.0)
```


**⭐ If you find this project helpful, please consider giving it a star!**
