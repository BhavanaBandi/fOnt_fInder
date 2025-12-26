#!/usr/bin/env python3
"""
Training script for Vision Transformer (ViT) and DeiT models.
"""

import argparse
import logging
from pathlib import Path

import torch

from train_utils import (
    set_seed, load_config, create_data_loaders,
    update_model_registry, setup_logging, save_class_names
)
from models.vit_model import ViTFontModel, DeiTFontModel, create_vit_model

logger = logging.getLogger(__name__)


def train_vit(args):
    """Train ViT/DeiT model."""
    config = load_config(args.config)
    
    set_seed(config['training'].get('seed', 42))
    
    setup_logging(config['logging']['log_dir'], f"vit_{args.variant}")
    
    logger.info(f"Training ViT model: {args.variant}")
    logger.info(f"Config: {config['training']}")
    
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
    
    if args.model_type == 'deit':
        model_name = f"deit_{args.variant}_patch16_224"
        model = DeiTFontModel(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=not args.from_scratch,
            drop_rate=config['models']['vit'].get('drop_rate', 0.1)
        )
    else:
        model = create_vit_model(
            variant=args.variant,
            num_classes=num_classes,
            pretrained=not args.from_scratch
        )
    
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    
    training_config = {
        'device': device,
        'epochs': args.epochs or config['training']['epochs'],
        'learning_rate': args.lr or config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'mixed_precision': config['training']['mixed_precision'],
        'early_stopping_patience': config['training']['early_stopping_patience']
    }
    
    results = model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        save_dir=config['registry']['checkpoints_dir']
    )
    
    logger.info(f"Training completed!")
    logger.info(f"Best accuracy: {results['best_accuracy']:.2f}%")
    logger.info(f"Top-5 accuracy: {results['final_top5_accuracy']:.2f}%")
    
    model_key = f"vit_{args.variant}" if args.model_type == 'vit' else f"deit_{args.variant}"
    update_model_registry(
        registry_path=config['registry']['path'],
        model_name=model_key,
        model_type='vit',
        checkpoint_path=str(Path(config['registry']['checkpoints_dir']) / f"{model.model_name}_best.pth"),
        accuracy=results['best_accuracy'],
        top5_accuracy=results['final_top5_accuracy'],
        num_params=model.get_num_parameters(),
        config={
            'variant': args.variant,
            'model_type': args.model_type,
            'num_classes': num_classes,
            'from_scratch': args.from_scratch
        }
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train ViT/DeiT model for font recognition')
    parser.add_argument('--config', type=str, default='./configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--variant', type=str, default='base',
                        choices=['tiny', 'small', 'base', 'large'],
                        help='ViT variant size')
    parser.add_argument('--model_type', type=str, default='vit',
                        choices=['vit', 'deit'],
                        help='Model type (ViT or DeiT)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--from_scratch', action='store_true',
                        help='Train from scratch without pretrained weights')
    
    args = parser.parse_args()
    train_vit(args)


if __name__ == '__main__':
    main()
