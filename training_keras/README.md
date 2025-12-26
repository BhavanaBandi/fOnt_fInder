# Keras/TensorFlow Training Scripts

**IMPORTANT**: These scripts require a **separate conda environment** with TensorFlow/Keras.

## Setup

```bash
# Create separate environment for Keras training
conda create -n font_detector_keras python=3.10
conda activate font_detector_keras

# Install TensorFlow/Keras
pip install tensorflow>=2.15.0 keras>=3.0.0

# Install other dependencies
pip install numpy pandas pillow matplotlib tqdm pyyaml
```

## Available Training Scripts

| Script | Model | Description |
|--------|-------|-------------|
| `train_efficientnet.py` | EfficientNetB0-B7 | Efficient CNN with compound scaling |
| `train_mobilenet.py` | MobileNetV2/V3 | Lightweight mobile-optimized models |
| `train_convnext.py` | ConvNeXt | Modern CNN architecture |

## Usage

```bash
# Activate Keras environment
conda activate font_detector_keras

# Train EfficientNet
python train_efficientnet.py --variant b0 --epochs 50

# Train MobileNetV3
python train_mobilenet.py --variant large --epochs 50

# Train ConvNeXt
python train_convnext.py --variant tiny --epochs 50
```

## Training from Scratch

All scripts support `--from_scratch` flag to train without pretrained weights:

```bash
python train_efficientnet.py --variant b0 --from_scratch --epochs 100
```

## Notes

- These scripts use Keras 3 API which is framework-agnostic
- Default backend is TensorFlow but can be changed via `KERAS_BACKEND` env var
- Mixed precision training is enabled by default for faster training on GPUs
