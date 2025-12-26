#!/usr/bin/env python3
"""
Training script for Hybrid CNN-Transformer models.
"""

import argparse
import logging
from pathlib import Path

import torch

from train_utils import (
    set_seed, load_config, create_data_loaders,
    update_model_registry, setup_logging, save_class_names
)
from models.hybrid_model import HybridFontModel, HybridEfficientNetFontModel, HybridResNetFontModel

logger = logging.getLogger(__name__)


def train_hybrid(args):
    """Train Hybrid CNN-Transformer model."""
    config = load_config(args.config)
    
    set_seed(config['training'].get('seed', 42))
    
    setup_logging(config['logging']['log_dir'], f"hybrid_{args.backbone}")
    
    logger.info(f"Training Hybrid model with backbone: {args.backbone}")
    
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
    
    hybrid_config = config['models']['hybrid']
    
    if 'efficientnet' in args.backbone:
        model = HybridEfficientNetFontModel(
            num_classes=num_classes,
            backbone_name=args.backbone,
            transformer_dim=hybrid_config.get('transformer_dim', 256),
            num_heads=hybrid_config.get('num_heads', 4),
            num_layers=hybrid_config.get('num_layers', 3),
            pretrained_backbone=not args.from_scratch
        )
    elif 'resnet' in args.backbone:
        model = HybridResNetFontModel(
            num_classes=num_classes,
            backbone_name=args.backbone,
            transformer_dim=hybrid_config.get('transformer_dim', 512),
            num_heads=hybrid_config.get('num_heads', 8),
            num_layers=hybrid_config.get('num_layers', 4),
            pretrained_backbone=not args.from_scratch
        )
    else:
        model = HybridFontModel(
            num_classes=num_classes,
            backbone_name=args.backbone,
            transformer_dim=hybrid_config.get('transformer_dim', 384),
            num_heads=hybrid_config.get('num_heads', 6),
            num_layers=hybrid_config.get('num_layers', 4),
            pretrained_backbone=not args.from_scratch,
            drop_rate=0.1
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
    
    update_model_registry(
        registry_path=config['registry']['path'],
        model_name=f"hybrid_{args.backbone}",
        model_type='hybrid',
        checkpoint_path=str(Path(config['registry']['checkpoints_dir']) / f"{model.model_name}_best.pth"),
        accuracy=results['best_accuracy'],
        top5_accuracy=results['final_top5_accuracy'],
        num_params=model.get_num_parameters(),
        config={
            'backbone': args.backbone,
            'transformer_dim': hybrid_config.get('transformer_dim', 384),
            'num_classes': num_classes,
            'from_scratch': args.from_scratch
        }
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid CNN-Transformer model')
    parser.add_argument('--config', type=str, default='./configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--backbone', type=str, default='convnext_tiny',
                        choices=['convnext_tiny', 'convnext_small', 'efficientnet_b0', 
                                'efficientnet_b2', 'resnet50', 'resnet101'],
                        help='CNN backbone architecture')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--from_scratch', action='store_true',
                        help='Train backbone from scratch')
    
    args = parser.parse_args()
    train_hybrid(args)


if __name__ == '__main__':
    main()
