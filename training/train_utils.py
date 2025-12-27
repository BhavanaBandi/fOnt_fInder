#!/usr/bin/env python3
"""
Training utilities: data loading, logging, and common functions.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import yaml

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = False):
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
        deterministic: If True, use deterministic algorithms (slower but reproducible)
                      If False, use faster non-deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # Slower but fully reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Faster - recommended for training
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms


def load_config(config_path: str = "./configs/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_train_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Get training augmentation transforms."""
    return transforms.Compose([
        transforms.Resize((int(image_size[0] * 1.15), int(image_size[1] * 1.15))),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.05),
        transforms.RandomRotation(degrees=3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)
    ])


def get_val_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Get validation/inference transforms."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def create_data_loaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 8,
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[DataLoader, DataLoader, int]:
    """Create training and validation data loaders with optimized settings."""
    
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    num_classes = len(train_dataset.classes)
    
    # Optimized DataLoader settings for faster training
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Number of classes: {num_classes}")
    
    return train_loader, val_loader, num_classes


def save_class_names(class_names: list, output_path: str):
    """Save class names to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(class_names, f, indent=2)


def update_model_registry(
    registry_path: str,
    model_name: str,
    model_type: str,
    checkpoint_path: str,
    accuracy: float,
    top5_accuracy: float,
    num_params: int,
    config: Dict
):
    """Update the model registry with a new trained model."""
    registry_path = Path(registry_path)
    
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = {}
    
    registry[model_name] = {
        'type': model_type,
        'checkpoint': checkpoint_path,
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'num_parameters': num_params,
        'num_classes': config.get('num_classes', 3812),
        'config': config
    }
    
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    logger.info(f"Updated registry with model: {model_name}")


class FontDataset(Dataset):
    """Custom dataset for font images with additional augmentations."""
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        max_samples_per_class: Optional[int] = None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.samples = []
        self.class_to_idx = {}
        
        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx
            
            images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
            
            if max_samples_per_class:
                images = images[:max_samples_per_class]
            
            for img_path in images:
                self.samples.append((str(img_path), idx))
        
        self.classes = list(self.class_to_idx.keys())
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def setup_logging(log_dir: str, model_name: str) -> logging.Logger:
    """Setup logging for training."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{model_name}_training.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)
