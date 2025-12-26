#!/usr/bin/env python3
"""
FasterViT-2 model implementation for font recognition.
Hierarchical attention with windowed and carrier tokens for efficiency.
Trained from scratch as per requirements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_model import FontModel


class PatchEmbed(nn.Module):
    """Patch embedding using convolutions."""
    
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 96,
        patch_size: int = 4
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class WindowAttention(nn.Module):
    """Windowed self-attention for local processing."""
    
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FasterViTBlock(nn.Module):
    """FasterViT block with hierarchical attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        use_carrier: bool = False,
        carrier_tokens: int = 4
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.use_carrier = use_carrier
        self.carrier_tokens = carrier_tokens
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size, num_heads,
            attn_drop=attn_drop, proj_drop=drop
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
        
        if use_carrier:
            self.carrier = nn.Parameter(torch.zeros(1, carrier_tokens, dim))
            self.carrier_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
            nn.init.trunc_normal_(self.carrier, std=0.02)
    
    def window_partition(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        Hp, Wp = x.shape[1], x.shape[2]
        
        x = x.view(B, Hp // self.window_size, self.window_size,
                   Wp // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, self.window_size * self.window_size, C)
        
        return windows, Hp, Wp
    
    def window_reverse(self, windows: torch.Tensor, H: int, W: int, Hp: int, Wp: int) -> torch.Tensor:
        B = int(windows.shape[0] / (Hp * Wp / self.window_size / self.window_size))
        
        x = windows.view(B, Hp // self.window_size, Wp // self.window_size,
                        self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, Hp, Wp, -1)
        
        if Hp > H or Wp > W:
            x = x[:, :H, :W, :].contiguous()
        
        return x.view(B, H * W, -1)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        
        windows, Hp, Wp = self.window_partition(x, H, W)
        attn_windows = self.attn(windows)
        x = self.window_reverse(attn_windows, H, W, Hp, Wp)
        
        x = shortcut + x
        
        if self.use_carrier:
            B = x.shape[0]
            carrier = self.carrier.expand(B, -1, -1)
            carrier, _ = self.carrier_attn(carrier, x, x)
            x = x + carrier.mean(dim=1, keepdim=True)
        
        x = x + self.mlp(self.norm2(x))
        
        return x


class FasterViTStage(nn.Module):
    """Single stage of FasterViT with downsampling."""
    
    def __init__(
        self,
        dim: int,
        out_dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        downsample: bool = True,
        use_carrier: bool = False
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            FasterViTBlock(
                dim, num_heads, window_size, mlp_ratio, drop,
                use_carrier=(use_carrier and i == depth - 1)
            )
            for i in range(depth)
        ])
        
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, out_dim),
            )
            self.downsample_conv = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        for block in self.blocks:
            x = block(x, H, W)
        
        if self.downsample is not None:
            B, L, C = x.shape
            x = x.view(B, H, W, C).permute(0, 3, 1, 2)
            x = self.downsample_conv(x)
            _, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
        
        return x, H, W


class FasterViTFontModel(FontModel):
    """
    FasterViT-2 style model for font recognition.
    Hierarchical architecture with windowed attention and carrier tokens.
    Trained from scratch.
    """
    
    def __init__(
        self,
        num_classes: int = 3812,
        in_channels: int = 3,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        use_carrier: bool = True
    ):
        super().__init__(num_classes, "fastervit2")
        
        self.num_stages = len(depths)
        dims = [embed_dim * (2 ** i) for i in range(self.num_stages)]
        
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size=4)
        
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage = FasterViTStage(
                dim=dims[i],
                out_dim=dims[i + 1] if i < self.num_stages - 1 else dims[i],
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                downsample=(i < self.num_stages - 1),
                use_carrier=(use_carrier and i >= self.num_stages - 2)
            )
            self.stages.append(stage)
        
        self.norm = nn.LayerNorm(dims[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(dims[-1], num_classes)
        
        self._model_config = {
            'architecture': 'fastervit2',
            'embed_dim': embed_dim,
            'depths': depths,
            'num_heads': num_heads,
            'window_size': window_size,
            'use_carrier': use_carrier
        }
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, H, W = self.patch_embed(x)
        
        for stage in self.stages:
            x, H, W = stage(x, H, W)
        
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2)).flatten(1)
        x = self.head(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        x, H, W = self.patch_embed(x)
        
        for stage in self.stages:
            x, H, W = stage(x, H, W)
        
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2)).flatten(1)
        
        return x


def create_fastervit_model(
    variant: str = "base",
    num_classes: int = 3812
) -> FasterViTFontModel:
    """Factory function for FasterViT models."""
    configs = {
        'tiny': {'embed_dim': 64, 'depths': [1, 1, 3, 1], 'num_heads': [2, 4, 8, 16]},
        'small': {'embed_dim': 80, 'depths': [2, 2, 4, 2], 'num_heads': [2, 4, 8, 16]},
        'base': {'embed_dim': 96, 'depths': [2, 2, 6, 2], 'num_heads': [3, 6, 12, 24]},
        'large': {'embed_dim': 128, 'depths': [2, 2, 8, 2], 'num_heads': [4, 8, 16, 32]},
    }
    
    config = configs.get(variant, configs['base'])
    return FasterViTFontModel(num_classes=num_classes, **config)
