#!/usr/bin/env python3
"""
Keras/TensorFlow training script for MobileNet models.

Based on official documentation:
- torchvision: https://github.com/pytorch/vision/blob/main/references/classification/README.md
- Keras: https://keras.io/api/applications/mobilenet/

Official MobileNetV3 training recommendations:
- RMSprop optimizer with momentum 0.9
- Learning rate: 0.064, decay by 0.973 every 2 epochs
- Epochs: 600
- Weight decay: 1e-5
- Dropout: 0.2
- AutoAugment (ImageNet policy)

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
from keras.applications import (
    MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large
)
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard


MOBILENET_MODELS = {
    'v1': (MobileNet, 224),
    'v2': (MobileNetV2, 224),
    'v3-small': (MobileNetV3Small, 224),
    'v3-large': (MobileNetV3Large, 224),
    'small': (MobileNetV3Small, 224),  # Alias
    'large': (MobileNetV3Large, 224),  # Alias
}


def create_data_augmentation(image_size):
    """Create data augmentation pipeline following official recommendations."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.02),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1),
        # RandomErasing equivalent
        layers.RandomCrop(int(image_size * 0.9), int(image_size * 0.9)),
        layers.Resizing(image_size, image_size),
    ], name="data_augmentation")


def create_model(variant, num_classes, from_scratch=False):
    """Create MobileNet model for font classification."""
    if variant not in MOBILENET_MODELS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(MOBILENET_MODELS.keys())}")
    
    model_class, image_size = MOBILENET_MODELS[variant]
    
    # Create base model
    base_model = model_class(
        include_top=False,
        weights=None if from_scratch else 'imagenet',
        input_shape=(image_size, image_size, 3),
        pooling='avg'
    )
    
    if not from_scratch:
        base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = create_data_augmentation(image_size)(inputs)
    
    # Rescale to [-1, 1] for MobileNet
    x = layers.Rescaling(1./127.5, offset=-1)(x)
    
    x = base_model(x, training=from_scratch)
    x = layers.Dropout(0.2)(x)  # Official dropout rate
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model, base_model, image_size


def create_datasets(train_dir, val_dir, image_size, batch_size):
    """Create training and validation datasets."""
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds


def train_mobilenet(args):
    """Train MobileNet model following official recommendations."""
    print(f"\n{'='*60}")
    print(f"Training MobileNet-{args.variant.upper()}")
    print(f"From scratch: {args.from_scratch}")
    print(f"{'='*60}\n")
    
    if args.mixed_precision:
        keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled")
    
    _, image_size = MOBILENET_MODELS[args.variant]
    
    train_ds, val_ds = create_datasets(
        args.train_dir,
        args.val_dir,
        image_size,
        args.batch_size
    )
    
    num_classes = len(train_ds.class_names)
    print(f"Number of classes: {num_classes}")
    
    # Save class names
    class_names_path = Path(args.save_dir) / 'class_names.json'
    class_names_path.parent.mkdir(parents=True, exist_ok=True)
    with open(class_names_path, 'w') as f:
        json.dump(train_ds.class_names, f)
    
    model, base_model, _ = create_model(args.variant, num_classes, args.from_scratch)
    
    # Official: RMSprop with lr=0.064, momentum=0.9
    # For fine-tuning, use lower LR
    if args.from_scratch:
        optimizer = keras.optimizers.RMSprop(
            learning_rate=0.064,
            momentum=0.9,
            rho=0.9
        )
    else:
        optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')
        ]
    )
    
    model.summary()
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Learning rate schedule: decay by 0.973 every 2 epochs (official)
    def lr_schedule(epoch, lr):
        if epoch > 0 and epoch % 2 == 0:
            return lr * 0.973
        return lr
    
    callbacks = [
        ModelCheckpoint(
            str(save_dir / f'mobilenet_{args.variant}_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=args.patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.LearningRateScheduler(lr_schedule) if args.from_scratch else 
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        TensorBoard(log_dir=str(save_dir / 'logs' / f'mobilenet_{args.variant}'))
    ]
    
    # Phase 1: Train with frozen base (if using pretrained)
    if not args.from_scratch:
        print("\n--- Phase 1: Training with frozen base ---")
        model.fit(
            train_ds,
            epochs=min(10, args.epochs),
            validation_data=val_ds,
            callbacks=callbacks
        )
        
        print("\n--- Phase 2: Fine-tuning entire model ---")
        base_model.trainable = True
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.lr / 10),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')
            ]
        )
    
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )
    
    model.save(str(save_dir / f'mobilenet_{args.variant}_final.keras'))
    
    best_acc = max(history.history['val_accuracy'])
    best_top5 = max(history.history['val_top5_accuracy'])
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_acc*100:.2f}%")
    print(f"Best Top-5 Accuracy: {best_top5*100:.2f}%")
    print(f"{'='*60}\n")
    
    return best_acc, best_top5


def main():
    parser = argparse.ArgumentParser(description='Train MobileNet with Keras/TensorFlow')
    parser.add_argument('--variant', type=str, default='v3-large',
                        choices=list(MOBILENET_MODELS.keys()),
                        help='MobileNet variant')
    parser.add_argument('--train_dir', type=str, 
                        default='./font_project_dataset/data/splits/train')
    parser.add_argument('--val_dir', type=str,
                        default='./font_project_dataset/data/splits/val')
    parser.add_argument('--save_dir', type=str, default='./models/checkpoints')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (official: 128)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Epochs (official: 600 for from-scratch)')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--from_scratch', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true', default=True)
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("KERAS/TENSORFLOW TRAINING - MOBILENET")
    print("Environment: font_detector_keras (NOT font_detector)")
    print("="*60 + "\n")
    
    train_mobilenet(args)


if __name__ == '__main__':
    main()
