#!/usr/bin/env python3
"""
Base FontModel class defining the standard interface for all font recognition models.
All models must conform to this interface for unified training and inference.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FontModel(ABC, nn.Module):
    """
    Abstract base class for font recognition models.
    All model implementations must inherit from this class.
    """
    
    def __init__(self, num_classes: int, model_name: str = "base"):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.training_history = []
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits of shape (batch, num_classes)."""
        pass
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str,
        scaler: Optional[torch.amp.GradScaler] = None,
        epoch: int = 0,
        total_epochs: int = 1
    ) -> Dict[str, float]:
        """Train for one epoch with progress bar."""
        self.train()
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
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{acc:.2f}%'
            })
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }
    
    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: str
    ) -> Dict[str, float]:
        """Validate the model."""
        self.eval()
        total_loss = 0.0
        correct = 0
        correct_top5 = 0
        total = 0
        
        pbar = tqdm(val_loader, desc="Validating", leave=False, ncols=100)
        
        with torch.no_grad():
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = self(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                _, top5_pred = outputs.topk(5, dim=1)
                correct_top5 += sum(labels[i] in top5_pred[i] for i in range(labels.size(0)))
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': 100. * correct / total,
            'top5_accuracy': 100. * correct_top5 / total
        }
    
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        save_dir: str = "./models/checkpoints"
    ) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dict
            save_dir: Directory to save checkpoints
            
        Returns:
            Training results dictionary
        """
        device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        epochs = config.get('epochs', 50)
        lr = config.get('learning_rate', 0.001)
        weight_decay = config.get('weight_decay', 0.0001)
        use_amp = config.get('mixed_precision', True)
        patience = config.get('early_stopping_patience', 10)
        
        self.to(device)
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        scaler = torch.amp.GradScaler('cuda') if use_amp and device == 'cuda' else None
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        best_acc = 0.0
        epochs_without_improvement = 0
        
        print(f"\n{'='*60}")
        print(f"Starting training: {epochs} epochs, LR={lr}, WD={weight_decay}")
        print(f"Train samples: {len(train_loader.dataset):,}, Val samples: {len(val_loader.dataset):,}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(
                train_loader, optimizer, criterion, device, scaler,
                epoch=epoch, total_epochs=epochs
            )
            
            val_metrics = self.validate(val_loader, criterion, device)
            
            scheduler.step()
            
            self.training_history.append({
                'epoch': epoch + 1,
                'train': train_metrics,
                'val': val_metrics,
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # Clear line and print epoch summary
            print(f"\n→ Epoch {epoch+1}/{epochs} Complete:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.2f}% | Top5: {val_metrics['top5_accuracy']:.2f}%")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}% - "
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, "
                f"Top5: {val_metrics['top5_accuracy']:.2f}%"
            )
            
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                epochs_without_improvement = 0
                
                self.save(
                    str(save_path / f"{self.model_name}_best.pth"),
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=val_metrics
                )
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        self.save(
            str(save_path / f"{self.model_name}_final.pth"),
            optimizer=optimizer,
            epoch=epochs,
            metrics=val_metrics
        )
        
        return {
            'best_accuracy': best_acc,
            'final_accuracy': val_metrics['accuracy'],
            'final_top5_accuracy': val_metrics['top5_accuracy'],
            'epochs_trained': epoch + 1,
            'history': self.training_history
        }
    
    def predict(self, image_tensor: torch.Tensor) -> Tuple[int, float]:
        """
        Predict class for a single image tensor.
        
        Args:
            image_tensor: Preprocessed image tensor (1, C, H, W)
            
        Returns:
            Tuple of (predicted_class_idx, confidence)
        """
        self.eval()
        with torch.no_grad():
            output = self(image_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred_idx = probs.max(dim=1)
        
        return pred_idx.item(), confidence.item()
    
    def predict_batch(
        self,
        image_tensors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict classes for a batch of images.
        
        Args:
            image_tensors: Batch of preprocessed images (B, C, H, W)
            
        Returns:
            Tuple of (predicted_indices, confidences)
        """
        self.eval()
        with torch.no_grad():
            outputs = self(image_tensors)
            probs = torch.softmax(outputs, dim=1)
            confidences, pred_indices = probs.max(dim=1)
        
        return pred_indices, confidences
    
    def save(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict] = None
    ):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'training_history': self.training_history
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load(self, path: str, device: str = 'cpu'):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        logger.info(f"Loaded checkpoint from {path}")
        
        return checkpoint
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> Dict[str, Any]:
        """Get model summary."""
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'num_parameters': self.get_num_parameters(),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
