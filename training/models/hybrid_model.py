#!/usr/bin/env python3
"""
Hybrid CNN-Transformer model for font recognition.
Uses CNN backbone (ConvNeXt/EfficientNet/ResNet) with Transformer encoder head.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Tuple
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_model import FontModel


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerHead(nn.Module):
    """Transformer encoder head for CNN features."""
    
    def __init__(
        self,
        in_features: int,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 4,
        num_classes: int = 3812,
        drop_rate: float = 0.1
    ):
        super().__init__()
        
        self.proj = nn.Linear(in_features, embed_dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim, num_heads,
                mlp_ratio=4.0,
                drop=drop_rate,
                attn_drop=drop_rate
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        if len(x.shape) == 4:
            x = x.flatten(2).transpose(1, 2)
        
        x = self.proj(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        for block in self.encoder_blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_output = x[:, 0]
        
        return self.head(cls_output)


class HybridFontModel(FontModel):
    """
    Hybrid CNN-Transformer model.
    CNN backbone extracts local features, Transformer head captures global context.
    """
    
    def __init__(
        self,
        num_classes: int = 3812,
        backbone_name: str = "convnext_tiny",
        transformer_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 4,
        pretrained_backbone: bool = True,
        drop_rate: float = 0.1
    ):
        super().__init__(num_classes, f"hybrid_{backbone_name}")
        
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained_backbone,
            features_only=True,
            out_indices=[-1]
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy)[-1]
            self.feature_dim = features.shape[1]
            self.feature_size = features.shape[2] * features.shape[3]
        
        self.transformer_head = TransformerHead(
            in_features=self.feature_dim,
            embed_dim=transformer_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            drop_rate=drop_rate
        )
        
        self._model_config = {
            'architecture': 'hybrid',
            'backbone': backbone_name,
            'transformer_dim': transformer_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'pretrained_backbone': pretrained_backbone
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)[-1]
        
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)
        
        return self.transformer_head(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get CNN backbone features."""
        return self.backbone(x)[-1]


class HybridEfficientNetFontModel(FontModel):
    """Hybrid model with EfficientNet backbone."""
    
    def __init__(
        self,
        num_classes: int = 3812,
        backbone_name: str = "efficientnet_b0",
        transformer_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        pretrained_backbone: bool = True
    ):
        super().__init__(num_classes, f"hybrid_{backbone_name}")
        
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained_backbone,
            features_only=True,
            out_indices=[-1]
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy)[-1]
            self.feature_dim = features.shape[1]
        
        self.transformer_head = TransformerHead(
            in_features=self.feature_dim,
            embed_dim=transformer_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)[-1]
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)
        return self.transformer_head(features)


class HybridResNetFontModel(FontModel):
    """Hybrid model with ResNet backbone."""
    
    def __init__(
        self,
        num_classes: int = 3812,
        backbone_name: str = "resnet50",
        transformer_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        pretrained_backbone: bool = True
    ):
        super().__init__(num_classes, f"hybrid_{backbone_name}")
        
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained_backbone,
            features_only=True,
            out_indices=[-1]
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy)[-1]
            self.feature_dim = features.shape[1]
        
        self.transformer_head = TransformerHead(
            in_features=self.feature_dim,
            embed_dim=transformer_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)[-1]
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)
        return self.transformer_head(features)


def create_hybrid_model(
    backbone: str = "convnext_tiny",
    num_classes: int = 3812,
    pretrained: bool = True
) -> FontModel:
    """Factory function to create hybrid models."""
    return HybridFontModel(
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained_backbone=pretrained
    )
