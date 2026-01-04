#!/usr/bin/env python3
"""Training script for FontNeXt (ConvNeXt multi-scale + Transformer head)."""

import argparse
import logging
from pathlib import Path

import torch

# ==========================================================================
# PERFORMANCE OPTIMIZATIONS - Enable BEFORE importing other torch modules
# ==========================================================================
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from train_utils import (
    set_seed, create_data_loaders, save_class_names, update_model_registry
)
from models.fontnext_model import FontNeXtFontModel

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_epoch(model, train_loader, optimizer, criterion, device, scaler, epoch, total_epochs):
    import sys
    from tqdm.auto import tqdm as tqdm_auto
    
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    num_batches = len(train_loader)

    pbar = tqdm_auto(
        train_loader,
        desc=f"Epoch {epoch+1}/{total_epochs}",
        file=sys.stderr,
        dynamic_ncols=True,
        leave=True,
        mininterval=0.5,
    )

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / (batch_idx + 1)
        acc = 100.0 * correct / total

        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.2f}%")

    return {
        'loss': total_loss / len(train_loader),
        'accuracy': 100.0 * correct / total,
    }


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    top5_correct = 0
    total = 0

    pbar = tqdm(val_loader, desc='Validating', leave=True, ncols=120)

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

            _, top5_pred = outputs.topk(5, dim=1)
            top5_correct += sum(labels[i] in top5_pred[i] for i in range(labels.size(0)))

            pbar.set_postfix({
                'loss': f'{total_loss / (pbar.n + 1):.4f}',
                'acc': f'{100.0 * correct / total:.2f}%',
            })

    return {
        'loss': total_loss / len(val_loader),
        'accuracy': 100.0 * correct / total,
        'top5_accuracy': 100.0 * top5_correct / total,
    }


def main():
    parser = argparse.ArgumentParser(description='Train FontNeXt (ConvNeXt multi-scale + Transformer)')
    parser.add_argument('--train_dir', type=str, default='./font_project_dataset/data/splits/train')
    parser.add_argument('--val_dir', type=str, default='./font_project_dataset/data/splits/val')
    parser.add_argument('--save_dir', type=str, default='./models/checkpoints')
    parser.add_argument('--registry_path', type=str, default='./models/registry.json')

    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_true', default=False)

    parser.add_argument('--patience', type=int, default=0, help='Early stopping patience (0=disabled)')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--from_scratch', action='store_true',
                        help='Train backbone from scratch (no pretrained weights)')

    # Model hyperparameters
    parser.add_argument('--backbone', type=str, default='convnext_tiny')
    parser.add_argument('--embed_dim', type=int, default=384)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--attn_drop_rate', type=float, default=0.1)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)

    parser.add_argument('--pool_sizes', type=int, nargs=4, default=[8, 8, 14, 7])

    parser.add_argument('--loss_fn', type=str, default='focal', choices=['focal', 'label_smoothing'])

    args = parser.parse_args()

    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = args.use_amp and not args.no_amp and device.type == 'cuda'

    print(f"\n{'='*70}")
    print("Training FontNeXt (ConvNeXt multi-scale + Transformer head)")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"{'='*70}\n")

    train_loader, val_loader, num_classes = create_data_loaders(
        args.train_dir,
        args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=(224, 224),
    )

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_class_names(
        train_loader.dataset.classes,
        str(Path(args.save_dir) / 'class_names.json'),
    )

    model = FontNeXtFontModel(
        num_classes=num_classes,
        backbone_name=args.backbone,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        pool_sizes=list(args.pool_sizes),
        pretrained_backbone=not args.from_scratch,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if args.loss_fn == 'focal':
        criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - args.warmup_epochs),
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )

    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    save_dir = Path(args.save_dir)

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            epoch,
            args.epochs,
        )

        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"\n→ Epoch {epoch+1}/{args.epochs} Complete:")
        print(f"  Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val Loss:   {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.2f}% | Top5: {val_metrics['top5_accuracy']:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}\n")

        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'num_classes': num_classes,
                'args': vars(args),
            }, save_dir / 'fontnext_best.pth')
            print(f"  ★ New best model saved! Acc: {best_acc:.2f}%")
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'num_classes': num_classes,
        'args': vars(args),
    }, save_dir / 'fontnext_final.pth')

    update_model_registry(
        registry_path=args.registry_path,
        model_name=f"fontnext_{args.backbone}",
        model_type='fontnext',
        checkpoint_path=str(save_dir / 'fontnext_best.pth'),
        accuracy=best_acc,
        top5_accuracy=val_metrics['top5_accuracy'],
        num_params=total_params,
        config={
            'backbone': args.backbone,
            'embed_dim': args.embed_dim,
            'depth': args.depth,
            'num_heads': args.num_heads,
            'mlp_ratio': args.mlp_ratio,
            'pool_sizes': list(args.pool_sizes),
            'from_scratch': args.from_scratch,
            'num_classes': num_classes,
        },
    )


if __name__ == '__main__':
    main()
