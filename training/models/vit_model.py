#!/usr/bin/env python3
"""
Vision Transformer (ViT) and DeiT models for font recognition.
Uses timm library for pretrained models and fine-tuning.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_model import FontModel


class ViTFontModel(FontModel):
    """
    Vision Transformer model for font classification.
    Pure patch-based transformer, fine-tuned from ImageNet.
    """
    
    def __init__(
        self,
        num_classes: int = 3812,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        drop_rate: float = 0.1
    ):
        super().__init__(num_classes, f"vit_{model_name}")
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate
        )
        
        self._model_config = {
            'architecture': 'vit',
            'base_model': model_name,
            'pretrained': pretrained,
            'drop_rate': drop_rate
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        return self.backbone.forward_features(x)


class DeiTFontModel(FontModel):
    """
    Data-efficient Image Transformer (DeiT) for font classification.
    Includes distillation token for knowledge distillation training.
    """
    
    def __init__(
        self,
        num_classes: int = 3812,
        model_name: str = "deit_base_patch16_224",
        pretrained: bool = True,
        drop_rate: float = 0.1
    ):
        super().__init__(num_classes, f"deit_{model_name}")
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate
        )
        
        self._model_config = {
            'architecture': 'deit',
            'base_model': model_name,
            'pretrained': pretrained,
            'drop_rate': drop_rate
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        return self.backbone.forward_features(x)


class ViTSmallFontModel(FontModel):
    """Smaller ViT variant for faster training/inference."""
    
    def __init__(
        self,
        num_classes: int = 3812,
        pretrained: bool = True,
        drop_rate: float = 0.1
    ):
        super().__init__(num_classes, "vit_small_patch16_224")
        
        self.backbone = timm.create_model(
            'vit_small_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ViTTinyFontModel(FontModel):
    """Tiny ViT variant for low-compute scenarios."""
    
    def __init__(
        self,
        num_classes: int = 3812,
        pretrained: bool = True,
        drop_rate: float = 0.1
    ):
        super().__init__(num_classes, "vit_tiny_patch16_224")
        
        self.backbone = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def create_vit_model(
    variant: str = "base",
    num_classes: int = 3812,
    pretrained: bool = True
) -> FontModel:
    """Factory function to create ViT models."""
    variants = {
        'tiny': 'vit_tiny_patch16_224',
        'small': 'vit_small_patch16_224',
        'base': 'vit_base_patch16_224',
        'large': 'vit_large_patch16_224'
    }
    
    model_name = variants.get(variant, 'vit_base_patch16_224')
    return ViTFontModel(num_classes, model_name, pretrained)
