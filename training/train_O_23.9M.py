#!/usr/bin/env python3
"""
Training script for Model_O_23.9M: ResNet-style CNN + Transformer (~23.9M parameters)

Architecture: ResNet-style CNN stem + PyTorch TransformerEncoder
Features: Lightest model, efficient training, tqdm progress bars

Usage:
    python training/train_O_23.9M.py --epochs 100 --batch_size 64 --lr 0.0005
"""

import argparse
import logging
import json
import os
from pathlib import Path
from datetime import datetime

import torch

# ============================================================================
# PERFORMANCE OPTIMIZATIONS - Enable BEFORE importing other torch modules
# ============================================================================
# TF32 (TensorFloat-32) for NVIDIA Ampere/Ada GPUs (RTX 30xx/40xx)
# Provides ~3x speedup with negligible precision loss
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from train_utils import (
    set_seed, create_data_loaders, setup_logging,
    save_class_names, update_model_registry
)
from models.Model_O_23_9M import FontClassifier

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', 
                                   label_smoothing=self.label_smoothing)
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_epoch(model, train_loader, optimizer, criterion, device, scaler, epoch, total_epochs):
    """Train for one epoch with tqdm progress bar and optimized data transfer."""
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
        # non_blocking=True allows async CPU->GPU transfer while GPU processes previous batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # set_to_none=True is faster than zeroing gradients
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            # Gradient clipping - critical for transformers
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Use .item() only for logging - it forces GPU sync but we need it for progress bar
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
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
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
    parser = argparse.ArgumentParser(description='Train Model_O_23.9M (ResNet + Transformer)')
    parser.add_argument('--train_dir', type=str, 
                        default='./font_project_dataset/data/splits/train')
    parser.add_argument('--val_dir', type=str,
                        default='./font_project_dataset/data/splits/val')
    parser.add_argument('--save_dir', type=str, default='./models/checkpoints')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (64 for 16GB GPU, lightest model)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Peak learning rate (default: 0.001 for from-scratch training)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision (default: enabled)')
    parser.add_argument('--no_amp', action='store_true', default=False,
                        help='Disable automatic mixed precision')
    parser.add_argument('--patience', type=int, default=0,
                        help='Early stopping patience (0=disabled)')
    parser.add_argument('--seed', type=int, default=42)
    
    # Model hyperparameters
    parser.add_argument('--transformer_layers', type=int, default=6)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Handle AMP flag
    use_amp = args.use_amp and not args.no_amp and device.type == 'cuda'
    
    print(f"\n{'='*70}")
    print(f"Training Model_O_23.9M (ResNet + Transformer, ~23.9M params)")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
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
    model = FontClassifier(
        num_classes=num_classes,
        transformer_layers=args.transformer_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function (Focal Loss for class imbalance)
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler with warmup - CRITICAL for transformer training
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.01,  # Start at 1% of LR
        end_factor=1.0, 
        total_iters=args.warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs - args.warmup_epochs,
        eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs]
    )
    
    # AMP scaler (enabled by default for faster training)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
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
    print(f"  AMP: {use_amp}")
    print(f"  Num Workers: {args.num_workers}")
    print(f"  Warmup Epochs: {args.warmup_epochs}")
    print(f"  Gradient Clipping: 1.0")
    print(f"  Early Stopping: {'Disabled' if args.patience == 0 else f'{args.patience} epochs'}")
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
            }, save_dir / 'model_O_23.9M_best.pth')
            print(f"  ★ New best model saved! Acc: {best_acc:.2f}%")
        else:
            patience_counter += 1
            # Early stopping only if patience > 0
            if args.patience > 0 and patience_counter >= args.patience:
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
    }, save_dir / 'model_O_23.9M_final.pth')
    
    # Save history
    with open(save_dir / 'model_O_23.9M_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {save_dir / 'model_O_23.9M_best.pth'}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
