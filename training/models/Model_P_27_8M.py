"""
Custom Hybrid CNN-ViT Model for Font Classification
Research-backed architecture combining EfficientNet stem with Vision Transformer encoder
Designed for 3,812 font classes with 764K training samples

Dataset: 224x224 RGB images
Architecture: Hybrid (CNN feature extraction + ViT reasoning)
Author: AI Engineer
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Tuple, List
import math


# ============================================================================
# BUILDING BLOCKS - Core Components
# ============================================================================

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck (MBConv) - From EfficientNet"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expansion_ratio: float = 6.0,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Expansion phase
        expanded_channels = int(in_channels * expansion_ratio)
        self.expand_conv = (
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
            if expansion_ratio != 1
            else nn.Identity()
        )
        self.expand_bn = nn.BatchNorm2d(expanded_channels)
        self.expand_act = nn.SiLU(inplace=True)
        
        # Depthwise convolution
        self.dw_conv = nn.Conv2d(
            expanded_channels,
            expanded_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=expanded_channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(expanded_channels)
        self.dw_act = nn.SiLU(inplace=True)
        
        # Squeeze-and-Excitation
        self.se_reduce = nn.Conv2d(expanded_channels, max(1, in_channels // 4), kernel_size=1)
        self.se_expand = nn.Conv2d(max(1, in_channels // 4), expanded_channels, kernel_size=1)
        
        # Projection phase
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip_connection = stride == 1 and in_channels == out_channels
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
    
    def forward(self, x):
        # Expansion
        out = self.expand_act(self.expand_bn(self.expand_conv(x)))
        
        # Depthwise
        out = self.dw_act(self.dw_bn(self.dw_conv(out)))
        
        # Squeeze-and-Excitation
        se = F.adaptive_avg_pool2d(out, 1)
        se = F.silu(self.se_reduce(se))
        se = torch.sigmoid(self.se_expand(se))
        out = out * se
        
        # Projection
        out = self.project_bn(self.project_conv(out))
        
        # Skip connection
        if self.skip_connection:
            out = out + self.dropout(x)
        
        return out


class PatchEmbedding(nn.Module):
    """Convert feature maps to patch embeddings for Transformer"""
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: [B, C, H, W] -> [B, embed_dim, H/patch_size, W/patch_size]
        x = self.proj(x)
        # -> [B, embed_dim, num_patches_h, num_patches_w]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention from Transformer"""
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout_rate: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout_rate: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads=num_heads, dropout_rate=dropout_rate)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout_rate),
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# CNN BACKBONE - EfficientNet-inspired Stem
# ============================================================================

class EfficientNetStem(nn.Module):
    """Lightweight CNN stem for local feature extraction"""
    def __init__(self, in_channels: int = 3, out_channels: int = 256, depth_multiplier: float = 1.0):
        super().__init__()
        
        base_channels = [32, 64, 128, 256]
        channels = [int(c * depth_multiplier) for c in base_channels]
        
        # Initial convolution
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(inplace=True),
        )
        
        # Stage 1: 112x112 -> 56x56
        self.stage1 = nn.Sequential(
            MBConvBlock(channels[0], channels[1], kernel_size=3, stride=2, expansion_ratio=6),
            MBConvBlock(channels[1], channels[1], kernel_size=3, stride=1, expansion_ratio=6),
        )
        
        # Stage 2: 56x56 -> 28x28
        self.stage2 = nn.Sequential(
            MBConvBlock(channels[1], channels[2], kernel_size=3, stride=2, expansion_ratio=6),
            MBConvBlock(channels[2], channels[2], kernel_size=3, stride=1, expansion_ratio=6),
            MBConvBlock(channels[2], channels[2], kernel_size=3, stride=1, expansion_ratio=6),
        )
        
        # Stage 3: 28x28 -> 14x14
        self.stage3 = nn.Sequential(
            MBConvBlock(channels[2], channels[3], kernel_size=3, stride=2, expansion_ratio=6),
            MBConvBlock(channels[3], channels[3], kernel_size=3, stride=1, expansion_ratio=6),
            MBConvBlock(channels[3], channels[3], kernel_size=3, stride=1, expansion_ratio=6),
        )
        
        # Feature adjustment layer
        self.feature_adjust = nn.Sequential(
            nn.Conv2d(channels[3], out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
    
    def forward(self, x):
        x = self.conv_stem(x)  # [B, 32, 112, 112]
        x = self.stage1(x)      # [B, 64, 56, 56]
        x = self.stage2(x)      # [B, 128, 28, 28]
        x = self.stage3(x)      # [B, 256, 14, 14]
        x = self.feature_adjust(x)  # [B, 256, 14, 14]
        return x


# ============================================================================
# MAIN MODEL - Hybrid CNN-ViT
# ============================================================================

class HybridCNNViT(nn.Module):
    """
    Hybrid CNN-Vision Transformer for Font Classification
    
    Architecture:
    1. CNN Stem (EfficientNet-inspired): Extract local features
    2. Patch Embedding: Convert feature maps to patch tokens
    3. Transformer Encoder: Global reasoning with self-attention
    4. Classification Head: Multi-class prediction
    
    Advantages:
    - CNN captures fine details (stroke thickness, serifs)
    - ViT captures global structure (overall font shape)
    - More efficient than pure ViT
    - More expressive than pure CNN
    """
    
    def __init__(
        self,
        num_classes: int = 3812,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_transformer_blocks: int = 8,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.1,
        cnn_depth_multiplier: float = 0.8,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Stage 1: CNN Stem for local feature extraction
        # Output: [B, 256, 14, 14] from 224x224 input
        cnn_out_channels = 256
        self.cnn_stem = EfficientNetStem(
            in_channels=3,
            out_channels=cnn_out_channels,
            depth_multiplier=cnn_depth_multiplier,
        )
        
        # Stage 2: Patch Embedding
        # Convert [B, 256, 14, 14] -> [B, 196, embed_dim] (14*14 = 196 patches)
        self.patch_embed = PatchEmbedding(cnn_out_channels, embed_dim, patch_size=1)
        
        # Classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings
        num_patches = 14 * 14  # 224 / 16 = 14, since CNN reduces 224->14
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout_rate)
        
        # Transformer encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_transformer_blocks)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim // 2, num_classes),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Stage 1: CNN feature extraction
        x = self.cnn_stem(x)  # [B, 256, 14, 14]
        
        # Stage 2: Patch embedding
        x = self.patch_embed(x)  # [B, 196, embed_dim]
        
        # Add classification token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 197, embed_dim]
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Stage 3: Transformer encoding
        for block in self.transformer_blocks:
            x = block(x)
        
        # Normalize
        x = self.norm(x)
        
        # Use CLS token for classification
        cls_output = x[:, 0]  # [B, embed_dim]
        
        # Stage 4: Classification head
        logits = self.head(cls_output)  # [B, num_classes]
        
        return logits


# ============================================================================
# LOSS FUNCTIONS - Handle Class Imbalance
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Reference: Lin et al. (2017) "Focal Loss for Dense Object Detection"
    
    Useful for your dataset which has imbalanced font classes
    Some fonts have 3600 samples, others have <1000
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets, class_weights=None):
        """
        Args:
            inputs: [B, num_classes] logits
            targets: [B] class indices
            class_weights: [num_classes] optional weights per class
        """
        log_p = F.log_softmax(inputs, dim=-1)
        ce = F.nll_loss(log_p, targets, weight=class_weights, reduction='none')
        p = torch.exp(-ce)
        loss = self.alpha * (1 - p) ** self.gamma * ce
        return loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing
    Prevents overconfidence on training set
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
    
    def forward(self, inputs, targets, class_weights=None):
        """
        Args:
            inputs: [B, num_classes] logits
            targets: [B] class indices
            class_weights: [num_classes] optional weights per class
        """
        log_p = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed target distribution
        target_dist = torch.zeros_like(log_p)
        target_dist.scatter_(1, targets.unsqueeze(1), 1.0)
        
        # Apply label smoothing
        target_dist = target_dist * (1 - self.smoothing) + self.smoothing / self.num_classes
        
        # KL divergence loss
        loss = -(target_dist * log_p).sum(dim=1)
        
        if class_weights is not None:
            loss = loss * class_weights[targets]
        
        return loss.mean()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_class_weights(class_counts: torch.Tensor, power: float = 0.5) -> torch.Tensor:
    """
    Compute weights for imbalanced classes
    
    Common strategies:
    1. Inverse frequency: w_i = 1 / count_i
    2. Balanced: w_i = (1 - power) / (1 + power) * (N / count_i)
    3. Smooth: w_i = 1 / sqrt(count_i)
    
    Args:
        class_counts: tensor of shape [num_classes] with sample counts
        power: weight power (0=uniform, 0.5=sqrt, 1.0=inverse)
    
    Returns:
        weights: tensor of shape [num_classes]
    """
    # Smooth inverse weighting
    weights = 1.0 / (class_counts.float() + 1) ** power
    weights = weights / weights.sum() * len(weights)
    return weights


def get_model(
    num_classes: int = 3812,
    pretrained: bool = False,
    **kwargs
) -> HybridCNNViT:
    """
    Factory function to create model
    
    Args:
        num_classes: number of font classes
        pretrained: load pretrained weights (if available)
        **kwargs: additional arguments to pass to model
    
    Returns:
        model: HybridCNNViT instance
    """
    model = HybridCNNViT(num_classes=num_classes, **kwargs)
    
    if pretrained:
        # Note: Pretrained weights would be loaded here
        # For now, we recommend training from scratch or using ImageNet-pretrained backbones
        pass
    
    return model


if __name__ == "__main__":
    # Test model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = HybridCNNViT(
        num_classes=3812,
        embed_dim=384,
        num_heads=6,
        num_transformer_blocks=8,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output[:2, :5]}")  # Print first 5 logits of first 2 samples
