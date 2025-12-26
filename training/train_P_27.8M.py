#!/usr/bin/env python3
"""
Training script for Model_P_27.8M: Hybrid CNN-ViT (~27.8M parameters)

Architecture: EfficientNet-inspired stem + Vision Transformer encoder
Features: FocalLoss, LabelSmoothing, tqdm progress bars

Usage:
    python training/train_P_27.8M.py --epochs 100 --batch_size 48 --lr 0.0005
"""

import argparse
import logging
import json
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from train_utils import (
    set_seed, create_data_loaders, setup_logging,
    save_class_names, update_model_registry
)
from models.Model_P_27_8M import (
    HybridCNNViT, FocalLoss, LabelSmoothingCrossEntropy, get_model
)

logger = logging.getLogger(__name__)


def train_epoch(model, train_loader, optimizer, criterion, device, scaler, epoch, total_epochs):
    """Train for one epoch with tqdm progress bar."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{total_epochs} [Train]",
        leave=True,
        ncols=120
    )
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / (batch_idx + 1)
        acc = 100. * correct / total
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.2f}%'})
    
    return {
        'loss': total_loss / len(train_loader),
        'accuracy': 100. * correct / total
    }


def validate(model, val_loader, criterion, device):
    """Validate with tqdm progress bar."""
    model.eval()
    total_loss = 0.0
    correct = 0
    top5_correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc="Validating", leave=True, ncols=120)
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, dim=1)
            top5_correct += sum(labels[i] in top5_pred[i] for i in range(labels.size(0)))
            
            pbar.set_postfix({
                'loss': f'{total_loss / (pbar.n + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    
    return {
        'loss': total_loss / len(val_loader),
        'accuracy': 100. * correct / total,
        'top5_accuracy': 100. * top5_correct / total
    }


def main():
    parser = argparse.ArgumentParser(description='Train Model_P_27.8M (Hybrid CNN-ViT)')
    parser.add_argument('--train_dir', type=str, 
                        default='./font_project_dataset/data/splits/train')
    parser.add_argument('--val_dir', type=str,
                        default='./font_project_dataset/data/splits/val')
    parser.add_argument('--save_dir', type=str, default='./models/checkpoints')
    parser.add_argument('--batch_size', type=int, default=48,
                        help='Batch size (48 for 16GB GPU, lighter model)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Use automatic mixed precision')
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    
    # Model hyperparameters
    parser.add_argument('--embed_dim', type=int, default=384)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--num_transformer_blocks', type=int, default=8)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    
    # Loss function choice
    parser.add_argument('--loss_fn', type=str, default='focal',
                        choices=['focal', 'label_smoothing'],
                        help='Loss function: focal or label_smoothing')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Training Model_P_27.8M (Hybrid CNN-ViT, ~27.8M params)")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Create data loaders
    train_loader, val_loader, num_classes = create_data_loaders(
        args.train_dir, args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=(224, 224)
    )
    
    print(f"Number of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    model = get_model(
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_transformer_blocks=args.num_transformer_blocks,
        dropout_rate=args.dropout_rate,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    if args.loss_fn == 'focal':
        criterion = FocalLoss(gamma=2.0)
    else:
        criterion = LabelSmoothingCrossEntropy(num_classes=num_classes, smoothing=0.1)
    
    print(f"Loss function: {args.loss_fn}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # AMP scaler
    scaler = torch.amp.GradScaler('cuda') if args.use_amp and device.type == 'cuda' else None
    
    # Training
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_acc = 0.0
    patience_counter = 0
    history = []
    
    print(f"\n{'='*70}")
    print(f"Training Configuration:")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  AMP: {args.use_amp}")
    print(f"{'='*70}\n")
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            epoch, args.epochs
        )
        
        val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # Log epoch results
        print(f"\n→ Epoch {epoch+1}/{args.epochs} Complete:")
        print(f"  Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val Loss:   {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.2f}% | Top5: {val_metrics['top5_accuracy']:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        history.append({
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'num_classes': num_classes,
                'args': vars(args)
            }, save_dir / 'model_P_27.8M_best.pth')
            print(f"  ★ New best model saved! Acc: {best_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'num_classes': num_classes,
        'args': vars(args)
    }, save_dir / 'model_P_27.8M_final.pth')
    
    # Save history
    with open(save_dir / 'model_P_27.8M_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {save_dir / 'model_P_27.8M_best.pth'}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
