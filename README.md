# 🔤 fOnt_fInder - Font Recognition System

End-to-end font detection and classification pipeline supporting multi-font images with **3,812 font classes**.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

This system provides:
- **Text Detection**: Locate text regions in images (using docTR/DBNet)
- **Font Classification**: Identify fonts from 3,812 font classes (764K images)
- **Multiple Architectures**: Custom Hybrid CNN-ViT, FasterViT, ViT, MobileNetV3, ConvNeXt
- **Web UI**: Streamlit-based interactive interface
- **Training from Scratch**: DeepFont-inspired training pipeline

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Images | 764,098 |
| Classes | 3,812 font variants |
| Font Families | 2,373 |
| Train/Val/Test | 80% / 10% / 10% |
| Image Size | 224×224 RGB |

## 📁 Project Structure

```
Font_Detector/
├── configs/
│   └── config.yaml              # Main configuration
├── pipeline/
│   ├── detection.py             # Text detection (docTR/DBNet)
│   ├── preprocessing.py         # Image preprocessing & augmentation
│   └── inference.py             # Unified inference engine
├── training/
│   ├── base_model.py            # FontModel base class with tqdm
│   ├── train_utils.py           # Data loading & utilities
│   ├── train_vit.py             # ViT/DeiT training
│   ├── train_hybrid.py          # Hybrid model training
│   ├── train_fastervit.py       # FasterViT training
│   ├── train_M_75M.py           # Custom Model M (75M params)
│   ├── train_P_27.8M.py         # Custom Model P (27.8M params)
│   ├── train_O_23.9M.py         # Custom Model O (23.9M params)
│   └── models/
│       ├── Model_M_75M.py       # Hybrid CNN-ViT (MBConv + Transformer)
│       ├── Model_P_27_8M.py     # EfficientNet stem + ViT
│       ├── Model_O_23_9M.py     # ResNet stem + Transformer
│       ├── vit_model.py         # Vision Transformer
│       ├── hybrid_model.py      # Hybrid CNN-Transformer
│       └── fastervit_model.py   # FasterViT implementation
├── training_keras/              # Keras/TensorFlow training (separate env)
│   ├── train_convnext.py        # ConvNeXt Keras training
│   ├── train_efficientnet.py    # EfficientNet Keras training
│   └── train_mobilenet.py       # MobileNet Keras training
├── models/
│   ├── registry.json            # Model registry
│   └── checkpoints/             # Saved model weights
├── ui/
│   └── app.py                   # Streamlit frontend
├── outputs/
│   ├── crops/                   # Detected text crops
│   ├── results/                 # Inference results
│   └── logs/                    # Training logs
└── font_project_dataset/        # Dataset (not in repo - 19GB)
    └── data/splits/{train,val,test}/
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create conda environment
conda create -n font_detector python=3.10
conda activate font_detector

# Install PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install timm albumentations opencv-python-headless pillow pandas tqdm pyyaml streamlit

# Optional: Install docTR for text detection
pip install python-doctr[torch]
```

### 2. Dataset Setup

Download the dataset and extract to `font_project_dataset/`:
```
font_project_dataset/
└── data/splits/
    ├── train/   # 611,278 images (80%)
    ├── val/     # 76,409 images (10%)
    └── test/    # 76,411 images (10%)
```

### 3. Train Models

#### Custom Models (Recommended)

```bash
# Model O - Lightest (23.9M params) - Best for quick experiments
python training/train_O_23.9M.py --epochs 100 --batch_size 64 --lr 0.0005

# Model P - Balanced (27.8M params)
python training/train_P_27.8M.py --epochs 100 --batch_size 48 --lr 0.0005

# Model M - Largest (75M params) - Best accuracy potential
python training/train_M_75M.py --epochs 100 --batch_size 32 --lr 0.0005
```

#### Pre-built Models

```bash
# FasterViT from scratch
python training/train_fastervit.py --variant small --epochs 100

# ViT/DeiT fine-tuning
python training/train_vit.py --variant base --epochs 50

# Hybrid CNN-Transformer
python training/train_hybrid.py --backbone convnext_tiny --epochs 50
```

### 3. Run Inference

```python
from pipeline.inference import run_inference

# Single image inference
results = run_inference(
    image_path="path/to/image.png",
    model_name="hybrid_convnext_tiny",
    detect_text=True
)

for r in results:
    print(f"Font: {r['font']}, Style: {r['style']}, Confidence: {r['confidence']:.2f}")
```

### 4. Launch UI

```bash
streamlit run ui/app.py
```

## 🏗️ Model Architectures

### Custom Models (Train from Scratch)

| Model | Parameters | Architecture | GPU Memory | Batch Size |
|-------|------------|--------------|------------|------------|
| **Model_O** | 23.9M | ResNet stem + Transformer | ~8GB | 64 |
| **Model_P** | 27.8M | EfficientNet stem + ViT | ~10GB | 48 |
| **Model_M** | 75M | MBConv + ViT (12 blocks) | ~14GB | 32 |

### Pre-built Models

| Model | Description | Training |
|-------|-------------|----------|
| **FasterViT** | Hierarchical attention, windowed tokens | From scratch |
| **ViT/DeiT** | Pure patch-based transformer | Fine-tune ImageNet |
| **Hybrid** | CNN backbone + Transformer head | Fine-tune |
| **MobileNetV3** | Mobile-optimized CNN | Fine-tune |
| **ConvNeXt** | Modern CNN architecture | Fine-tune |

## Inference API

```python
from pipeline.inference import FontInferenceEngine

engine = FontInferenceEngine(
    model_registry_path="./models/registry.json",
    default_model="hybrid_convnext_tiny"
)

results = engine.run_inference(
    image_path="image.png",
    model_name="hybrid_convnext_tiny",
    detect_text=True,           # Run text detection
    single_text_mode=False      # False for multi-font images
)

# Result format
# [
#   {
#     "bbox": [x1, y1, x2, y2],
#     "font": "Roboto",
#     "style": "Bold",
#     "confidence": 0.87,
#     "model_used": "hybrid_convnext_tiny"
#   }
# ]
```

## Dataset

The dataset contains 764,098 images across 3,812 font classes:
- **Source**: Google Fonts (rendered synthetic images)
- **Split**: 80% train / 10% val / 10% test
- **Augmentations**: Perspective, blur, noise, compression

## Model Registry

Models are registered in `models/registry.json`:

```json
{
  "model_name": {
    "type": "vit | hybrid | fastervit | lightweight",
    "checkpoint": "path/to/checkpoint.pth",
    "accuracy": 85.5,
    "top5_accuracy": 95.2,
    "num_parameters": 28000000,
    "config": {...}
  }
}
```

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  mixed_precision: true

preprocessing:
  image_size: [224, 224]
  normalize_contrast: true

detection:
  model: "db_resnet50"
  min_confidence: 0.5
```

## ⚡ Performance Tips

1. **GPU Training**: Use CUDA for 10-20x speedup
2. **Batch Size**: Adjust based on GPU memory (see model table above)
3. **Learning Rate**: 0.0005 recommended for from-scratch training
4. **Early Stopping**: Patience of 15 epochs to prevent overfitting
5. **Focal Loss**: Used to handle class imbalance (some fonts have 3600 samples, others <1000)

## 🔧 Training Hyperparameters

Based on DeepFont paper, adapted for 3x larger dataset:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | Weight decay 0.02 |
| LR Schedule | Cosine + Warmup | 5 epoch warmup |
| Learning Rate | 0.0005 | Lower for large dataset |
| Label Smoothing | 0.1 | Prevents overconfidence |
| Loss Function | Focal Loss (γ=2) | Handles class imbalance |

## 👥 Contributors

- Training Pipeline & Custom Models

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- [DeepFont](https://research.adobe.com/publication/deepfont-identify-your-font-from-an-image/) by Adobe Research
- [Google Fonts](https://fonts.google.com/) for font data
- [timm](https://github.com/huggingface/pytorch-image-models) for pretrained models
