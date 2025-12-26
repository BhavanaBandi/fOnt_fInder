#!/usr/bin/env python3
"""
Training script for FasterViT-2 model (trained from scratch).

Based on NVlabs/FasterViT official repository:
https://github.com/NVlabs/FasterViT

Official training uses:
- MESA training technique
- AdamW or LAMB optimizer  
- LR: 0.005, Weight decay: 0.005-0.12
- Drop path rate: 0.2-0.5
- AMP and EMA enabled
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn

from train_utils import (
    set_seed, load_config, create_data_loaders,
    update_model_registry, setup_logging, save_class_names
)

# Try to use official FasterViT package first, fallback to custom implementation
try:
    from fastervit import create_model as create_fastervit_official
    HAS_OFFICIAL_FASTERVIT = True
except ImportError:
    HAS_OFFICIAL_FASTERVIT = False
    
from models.fastervit_model import FasterViTFontModel, create_fastervit_model
from base_model import FontModel

logger = logging.getLogger(__name__)


class OfficialFasterViTWrapper(FontModel):
    """Wrapper for official FasterViT that inherits FontModel training interface."""
    
    def __init__(self, model, num_classes, model_name):
        super().__init__(num_classes, model_name)
        self.backbone = model
        self._model_config = {'source': 'official_nvlabs_fastervit'}
    
    def forward(self, x):
        return self.backbone(x)


def train_fastervit(args):
    """Train FasterViT model from scratch using official NVlabs implementation."""
    config = load_config(args.config)
    
    set_seed(config['training'].get('seed', 42))
    
    setup_logging(config['logging']['log_dir'], f"fastervit_{args.variant}")
    
    logger.info(f"Training FasterViT-2 model variant: {args.variant}")
    logger.info("Training from scratch (no pretrained weights)")
    
    if HAS_OFFICIAL_FASTERVIT:
        logger.info("Using OFFICIAL NVlabs FasterViT implementation")
    else:
        logger.info("Official fastervit package not found, using custom implementation")
        logger.info("Install official: pip install fastervit")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    train_loader, val_loader, num_classes = create_data_loaders(
        train_dir=config['dataset']['train_dir'],
        val_dir=config['dataset']['val_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        image_size=tuple(config['preprocessing']['image_size'])
    )
    
    save_class_names(
        train_loader.dataset.classes,
        str(Path(config['registry']['checkpoints_dir']) / 'class_names.json')
    )
    
    # Use official FasterViT if available
    if HAS_OFFICIAL_FASTERVIT and not args.use_custom:
        # Official FasterViT model names
        variant_map = {
            'tiny': 'faster_vit_0_224',
            'small': 'faster_vit_1_224', 
            'base': 'faster_vit_2_224',
            'large': 'faster_vit_3_224'
        }
        model_name = variant_map.get(args.variant, 'faster_vit_2_224')
        
        # Create without pretrained weights (training from scratch)
        official_model = create_fastervit_official(
            model_name,
            pretrained=False,
            num_classes=num_classes
        )
        model = OfficialFasterViTWrapper(official_model, num_classes, model_name)
    else:
        # Fallback to custom implementation
        model = create_fastervit_model(
            variant=args.variant,
            num_classes=num_classes
        )
    
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    
    # Official FasterViT training hyperparameters from TRAINING.md
    epochs = args.epochs or 300  # Official uses 300 epochs
    lr = args.lr or 0.005  # Official LR
    weight_decay = 0.005  # Official weight decay for smaller variants
    drop_path = 0.2  # Official drop path rate
    
    if args.variant in ['large']:
        weight_decay = 0.12  # Higher for larger models
        drop_path = 0.3
    
    training_config = {
        'device': device,
        'epochs': epochs,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'mixed_precision': True,  # Official uses AMP
        'early_stopping_patience': 30,
        'use_ema': True  # Official uses EMA
    }
    
    logger.info(f"Training config (following NVlabs official):")
    logger.info(f"  epochs={epochs}, lr={lr}, wd={weight_decay}, drop_path={drop_path}")
    logger.info(f"  AMP=True, EMA=True (official settings)")
    
    results = model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        save_dir=config['registry']['checkpoints_dir']
    )
    
    logger.info(f"Training completed!")
    logger.info(f"Best accuracy: {results['best_accuracy']:.2f}%")
    logger.info(f"Top-5 accuracy: {results['final_top5_accuracy']:.2f}%")
    
    update_model_registry(
        registry_path=config['registry']['path'],
        model_name=f"fastervit_{args.variant}",
        model_type='fastervit',
        checkpoint_path=str(Path(config['registry']['checkpoints_dir']) / f"{model.model_name}_best.pth"),
        accuracy=results['best_accuracy'],
        top5_accuracy=results['final_top5_accuracy'],
        num_params=model.get_num_parameters(),
        config={
            'variant': args.variant,
            'num_classes': num_classes,
            'trained_from_scratch': True,
            **model._model_config
        }
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train FasterViT-2 model from scratch')
    parser.add_argument('--config', type=str, default='./configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--variant', type=str, default='base',
                        choices=['tiny', 'small', 'base', 'large'],
                        help='FasterViT variant (tiny=0, small=1, base=2, large=3)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs (official: 300)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate (official: 0.005)')
    parser.add_argument('--use_custom', action='store_true',
                        help='Use custom implementation instead of official')
    
    args = parser.parse_args()
    
    if not HAS_OFFICIAL_FASTERVIT:
        print("\n" + "="*60)
        print("NOTE: Official FasterViT package not installed.")
        print("Install with: pip install fastervit")
        print("Using custom implementation as fallback.")
        print("="*60 + "\n")
    
    train_fastervit(args)


if __name__ == '__main__':
    main()
