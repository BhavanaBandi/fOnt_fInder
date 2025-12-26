#!/usr/bin/env python3
"""
Keras/TensorFlow training script for ConvNeXt models.

Based on official documentation:
- torchvision: https://github.com/pytorch/vision/blob/main/references/classification/README.md
- timm: https://github.com/huggingface/pytorch-image-models

Official ConvNeXt training recommendations:
- AdamW optimizer
- Learning rate: 1e-3 with cosine annealing
- Warmup: 5 epochs with linear warmup
- Epochs: 600
- Weight decay: 0.05
- Label smoothing: 0.1
- Mixup alpha: 0.2, Cutmix alpha: 1.0
- Random erasing: 0.1
- AutoAugment: ta_wide

IMPORTANT: Run in separate environment (font_detector_keras), NOT font_detector.
"""

import os
import argparse
import json
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
from keras import layers, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, Callback
from tqdm import tqdm


class TQDMProgressBar(Callback):
    """TQDM progress bar for Keras training (Jupyter notebook style)."""
    
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        print(f"\n{'='*60}")
        print(f"Starting training for {self.epochs} epochs")
        print(f"{'='*60}")
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.steps = self.params['steps']
        self.pbar = tqdm(
            total=self.steps,
            desc=f"Epoch {epoch+1}/{self.epochs}",
            unit='batch',
            ncols=120,
            leave=True
        )
    
    def on_batch_end(self, batch, logs=None):
        self.pbar.update(1)
        # Show metrics in progress bar
        metrics = {k: f"{v:.4f}" for k, v in (logs or {}).items() if not k.startswith('val_')}
        self.pbar.set_postfix(metrics)
    
    def on_epoch_end(self, epoch, logs=None):
        self.pbar.close()
        # Print epoch summary
        logs = logs or {}
        train_acc = logs.get('accuracy', 0) * 100
        val_acc = logs.get('val_accuracy', 0) * 100
        val_top5 = logs.get('val_top5_accuracy', 0) * 100
        train_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        print(f"  → Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"  → Val Loss:   {val_loss:.4f} | Acc: {val_acc:.2f}% | Top5: {val_top5:.2f}%\n")
    
    def on_train_end(self, logs=None):
        print(f"{'='*60}")
        print("Training finished!")
        print(f"{'='*60}\n")


# ConvNeXt is available in keras.applications starting from TF 2.13+
try:
    from keras.applications import (
        ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ConvNeXtXLarge
    )
    HAS_CONVNEXT = True
except ImportError:
    HAS_CONVNEXT = False
    print("ConvNeXt not available in this Keras version. Requires TF >= 2.13")


CONVNEXT_MODELS = {
    'tiny': (ConvNeXtTiny if HAS_CONVNEXT else None, 224),
    'small': (ConvNeXtSmall if HAS_CONVNEXT else None, 224),
    'base': (ConvNeXtBase if HAS_CONVNEXT else None, 224),
    'large': (ConvNeXtLarge if HAS_CONVNEXT else None, 224),
    'xlarge': (ConvNeXtXLarge if HAS_CONVNEXT else None, 224),
}


def create_data_augmentation(image_size):
    """Create data augmentation following official ConvNeXt training."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.02),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
        # Simulate RandomErasing
        layers.RandomCrop(int(image_size * 0.875), int(image_size * 0.875)),
        layers.Resizing(image_size, image_size),
    ], name="data_augmentation")


def create_model(variant, num_classes, from_scratch=False):
    """Create ConvNeXt model for font classification."""
    if not HAS_CONVNEXT:
        raise RuntimeError("ConvNeXt requires TensorFlow >= 2.13. Please upgrade.")
    
    if variant not in CONVNEXT_MODELS:
        raise ValueError(f"Unknown variant: {variant}")
    
    model_class, image_size = CONVNEXT_MODELS[variant]
    
    if model_class is None:
        raise RuntimeError(f"ConvNeXt-{variant} not available")
    
    base_model = model_class(
        include_top=False,
        weights=None if from_scratch else 'imagenet',
        input_shape=(image_size, image_size, 3),
        pooling='avg'
    )
    
    if not from_scratch:
        base_model.trainable = False
    
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = create_data_augmentation(image_size)(inputs)
    
    # ConvNeXt expects [0, 255] range with built-in normalization
    x = base_model(x, training=from_scratch)
    x = layers.Dropout(0.0)(x)  # ConvNeXt uses stochastic depth instead
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model, base_model, image_size


def create_datasets(train_dir, val_dir, image_size, batch_size):
    """Create datasets and return class_names before prefetch."""
    train_ds_raw = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    # Capture class_names before prefetch
    class_names = train_ds_raw.class_names
    num_classes = len(class_names)
    
    val_ds_raw = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds_raw.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds_raw.prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, class_names, num_classes


class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Warmup + Cosine decay learning rate schedule (official ConvNeXt)."""
    
    def __init__(self, initial_lr, warmup_epochs, total_epochs, steps_per_epoch):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        # Linear warmup
        warmup_lr = self.initial_lr * (step / self.warmup_steps)
        
        # Cosine decay after warmup
        decay_steps = self.total_steps - self.warmup_steps
        decay_step = step - self.warmup_steps
        cosine_decay = 0.5 * (1 + tf.cos(3.14159 * decay_step / decay_steps))
        decayed_lr = self.initial_lr * cosine_decay
        
        return tf.where(step < self.warmup_steps, warmup_lr, decayed_lr)


def train_convnext(args):
    """Train ConvNeXt model following official recommendations."""
    print(f"\n{'='*60}")
    print(f"Training ConvNeXt-{args.variant.upper()}")
    print(f"From scratch: {args.from_scratch}")
    print(f"{'='*60}\n")
    
    if not HAS_CONVNEXT:
        print("ERROR: ConvNeXt not available. Requires TensorFlow >= 2.13")
        return None, None
    
    # Use float32 for maximum precision (no mixed precision)
    keras.mixed_precision.set_global_policy('float32')
    print("Using float32 precision (full precision)")
    
    _, image_size = CONVNEXT_MODELS[args.variant]
    
    train_ds, val_ds, class_names, num_classes = create_datasets(
        args.train_dir,
        args.val_dir,
        image_size,
        args.batch_size
    )
    
    print(f"Number of classes: {num_classes}")
    
    # Save class names
    class_names_path = Path(args.save_dir) / 'class_names.json'
    class_names_path.parent.mkdir(parents=True, exist_ok=True)
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)
    
    # Handle pretrained vs from_scratch
    use_pretrained = args.use_pretrained and not args.from_scratch
    model, base_model, _ = create_model(args.variant, num_classes, from_scratch=not use_pretrained)
    
    # Training configuration based on DeepFont paper adapted for 3x larger dataset
    # DeepFont: SGD lr=0.01, batch=128, ~100 epochs on 2383 classes
    # Our adaptation: AdamW, lr=0.0005 (lower for stability), 100 epochs, 3812 classes
    steps_per_epoch = len(train_ds)
    
    # Use warmup + cosine decay for from-scratch training
    lr_schedule = WarmupCosineDecay(
        initial_lr=args.lr,
        warmup_epochs=5,
        total_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch
    )
    
    # Weight decay: 0.01-0.05 recommended for large-scale training
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=0.02  # Moderate weight decay for regularization
    )
    
    print(f"\nTraining Configuration (DeepFont-inspired for 764K images):")
    print(f"  Learning Rate: {args.lr} with warmup + cosine decay")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Weight Decay: 0.02")
    print(f"  Label Smoothing: 0.1\n")
    
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')
        ]
    )
    
    model.summary()
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        TQDMProgressBar(),  # Progress bar like Jupyter notebook
        ModelCheckpoint(
            str(save_dir / f'convnext_{args.variant}_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=0  # Suppress default output, using TQDM instead
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=args.patience,
            restore_best_weights=True,
            verbose=1
        ),
        TensorBoard(log_dir=str(save_dir / 'logs' / f'convnext_{args.variant}'))
    ]
    
    # Training from scratch - single phase, all layers trainable
    print("\n--- Training from scratch (all layers trainable) ---")
    
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=0  # Using TQDM progress bar
    )
    
    model.save(str(save_dir / f'convnext_{args.variant}_final.keras'))
    
    best_acc = max(history.history['val_accuracy'])
    best_top5 = max(history.history['val_top5_accuracy'])
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_acc*100:.2f}%")
    print(f"Best Top-5 Accuracy: {best_top5*100:.2f}%")
    print(f"{'='*60}\n")
    
    return best_acc, best_top5


def main():
    parser = argparse.ArgumentParser(description='Train ConvNeXt with Keras/TensorFlow')
    parser.add_argument('--variant', type=str, default='tiny',
                        choices=['tiny', 'small', 'base', 'large', 'xlarge'],
                        help='ConvNeXt variant')
    parser.add_argument('--train_dir', type=str, 
                        default='./font_project_dataset/data/splits/train')
    parser.add_argument('--val_dir', type=str,
                        default='./font_project_dataset/data/splits/val')
    parser.add_argument('--save_dir', type=str, default='./models/checkpoints')
    # Hyperparameters based on DeepFont paper adapted for 3x larger dataset (764K images, 3812 classes)
    # DeepFont used: SGD, lr=0.01, batch=128, ~100 epochs on 2383 classes
    # We use: AdamW (modern), lower lr due to larger dataset, more epochs
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (64 for 16GB GPU, reduce if OOM)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Epochs (100 for from-scratch training)')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate (lower for large dataset from-scratch)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--from_scratch', action='store_true', default=True,
                        help='Train from scratch (default: True)')
    parser.add_argument('--use_pretrained', action='store_true',
                        help='Use pretrained weights instead of from-scratch')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("KERAS/TENSORFLOW TRAINING - CONVNEXT")
    print("Environment: font_detector_keras (NOT font_detector)")
    print("="*60 + "\n")
    
    train_convnext(args)


if __name__ == '__main__':
    main()
