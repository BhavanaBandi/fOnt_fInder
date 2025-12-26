#!/usr/bin/env python3
"""
Keras/TensorFlow training script for EfficientNet models.

Based on official Keras documentation:
https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning

IMPORTANT: Run in separate environment (font_detector_keras), NOT font_detector.

Official EfficientNet training recommendations:
- Use RMSprop optimizer with decay 0.9, momentum 0.9
- Learning rate: 0.256, decayed by 0.97 every 2.4 epochs
- AutoAugment for data augmentation
- Stochastic depth with survival probability 0.8
- Label smoothing 0.1
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
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
    EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3,
    EfficientNetV2S, EfficientNetV2M, EfficientNetV2L
)
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard


EFFICIENTNET_MODELS = {
    'b0': (EfficientNetB0, 224),
    'b1': (EfficientNetB1, 240),
    'b2': (EfficientNetB2, 260),
    'b3': (EfficientNetB3, 300),
    'b4': (EfficientNetB4, 380),
    'b5': (EfficientNetB5, 456),
    'b6': (EfficientNetB6, 528),
    'b7': (EfficientNetB7, 600),
    'v2-b0': (EfficientNetV2B0, 224),
    'v2-b1': (EfficientNetV2B1, 240),
    'v2-b2': (EfficientNetV2B2, 260),
    'v2-b3': (EfficientNetV2B3, 300),
    'v2-s': (EfficientNetV2S, 384),
    'v2-m': (EfficientNetV2M, 480),
    'v2-l': (EfficientNetV2L, 480),
}


def create_data_augmentation(image_size):
    """Create data augmentation pipeline."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.02),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1),
    ], name="data_augmentation")


def create_model(variant, num_classes, from_scratch=False):
    """Create EfficientNet model for font classification."""
    if variant not in EFFICIENTNET_MODELS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(EFFICIENTNET_MODELS.keys())}")
    
    model_class, image_size = EFFICIENTNET_MODELS[variant]
    
    # Create base model
    base_model = model_class(
        include_top=False,
        weights=None if from_scratch else 'imagenet',
        input_shape=(image_size, image_size, 3),
        pooling='avg'
    )
    
    if not from_scratch:
        # Freeze base model for initial training
        base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = create_data_augmentation(image_size)(inputs)
    
    # EfficientNet expects inputs in [0, 255] range
    # No rescaling needed as EfficientNet has built-in preprocessing
    
    x = base_model(x, training=from_scratch)
    x = layers.Dropout(0.3)(x)
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
    
    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds


def train_efficientnet(args):
    """Train EfficientNet model."""
    print(f"\n{'='*60}")
    print(f"Training EfficientNet-{args.variant.upper()}")
    print(f"From scratch: {args.from_scratch}")
    print(f"{'='*60}\n")
    
    # Enable mixed precision for faster training
    if args.mixed_precision:
        keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled")
    
    # Get model info
    _, image_size = EFFICIENTNET_MODELS[args.variant]
    
    # Create datasets
    train_ds, val_ds = create_datasets(
        args.train_dir,
        args.val_dir,
        image_size,
        args.batch_size
    )
    
    # Get number of classes
    num_classes = len(train_ds.class_names)
    print(f"Number of classes: {num_classes}")
    
    # Save class names
    class_names_path = Path(args.save_dir) / 'class_names.json'
    class_names_path.parent.mkdir(parents=True, exist_ok=True)
    with open(class_names_path, 'w') as f:
        json.dump(train_ds.class_names, f)
    
    # Create model
    model, base_model, _ = create_model(args.variant, num_classes, args.from_scratch)
    
    # Compile with official recommendations
    # Official: RMSprop with lr=0.256, decay 0.97 every 2.4 epochs
    # We use Adam for simplicity with lower LR for fine-tuning
    initial_lr = args.lr if args.from_scratch else args.lr / 10
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')
        ]
    )
    
    model.summary()
    
    # Callbacks
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            str(save_dir / f'efficientnet_{args.variant}_best.keras'),
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
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=str(save_dir / 'logs' / f'efficientnet_{args.variant}')
        )
    ]
    
    # Phase 1: Train with frozen base (if using pretrained)
    if not args.from_scratch:
        print("\n--- Phase 1: Training with frozen base ---")
        history1 = model.fit(
            train_ds,
            epochs=min(10, args.epochs),
            validation_data=val_ds,
            callbacks=callbacks
        )
        
        # Phase 2: Fine-tune entire model
        print("\n--- Phase 2: Fine-tuning entire model ---")
        base_model.trainable = True
        
        # Use lower learning rate for fine-tuning
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=initial_lr / 10),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')
            ]
        )
    
    # Train
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )
    
    # Save final model
    model.save(str(save_dir / f'efficientnet_{args.variant}_final.keras'))
    
    # Get best metrics
    best_acc = max(history.history['val_accuracy'])
    best_top5 = max(history.history['val_top5_accuracy'])
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_acc*100:.2f}%")
    print(f"Best Top-5 Accuracy: {best_top5*100:.2f}%")
    print(f"Model saved to: {save_dir}")
    print(f"{'='*60}\n")
    
    return best_acc, best_top5


def main():
    parser = argparse.ArgumentParser(description='Train EfficientNet with Keras/TensorFlow')
    parser.add_argument('--variant', type=str, default='b0',
                        choices=list(EFFICIENTNET_MODELS.keys()),
                        help='EfficientNet variant')
    parser.add_argument('--train_dir', type=str, 
                        default='./font_project_dataset/data/splits/train',
                        help='Training data directory')
    parser.add_argument('--val_dir', type=str,
                        default='./font_project_dataset/data/splits/val',
                        help='Validation data directory')
    parser.add_argument('--save_dir', type=str, default='./models/checkpoints',
                        help='Directory to save models')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--from_scratch', action='store_true',
                        help='Train from scratch without pretrained weights')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                        help='Use mixed precision training')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("KERAS/TENSORFLOW TRAINING SCRIPT")
    print("Make sure you're in the 'font_detector_keras' environment!")
    print("="*60 + "\n")
    
    train_efficientnet(args)


if __name__ == '__main__':
    main()
