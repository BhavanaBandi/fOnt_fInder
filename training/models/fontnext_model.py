import torch
import torch.nn as nn
import timm
from timm.layers import DropPath
from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_model import FontModel


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

        if layer_scale_init_value > 0:
            self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
            self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        if self.gamma_1 is not None:
            x = x + self.drop_path(attn_out * self.gamma_1)
        else:
            x = x + self.drop_path(attn_out)

        mlp_out = self.mlp(self.norm2(x))
        if self.gamma_2 is not None:
            x = x + self.drop_path(mlp_out * self.gamma_2)
        else:
            x = x + self.drop_path(mlp_out)

        return x


class FontNeXtFontModel(FontModel):
    def __init__(
        self,
        num_classes: int = 3812,
        backbone_name: str = "convnext_tiny",
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        pool_sizes: Optional[List[int]] = None,
        pretrained_backbone: bool = True,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__(num_classes, f"fontnext_{backbone_name}")

        if pool_sizes is None:
            pool_sizes = [8, 8, 14, 7]

        if len(pool_sizes) != 4:
            raise ValueError("pool_sizes must have length 4")

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained_backbone,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        feature_channels = list(self.backbone.feature_info.channels())
        self.num_scales = len(feature_channels)

        if self.num_scales != 4:
            raise ValueError("Backbone must return 4 feature maps")

        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((p, p)) for p in pool_sizes
        ])
        self.projs = nn.ModuleList([
            nn.Conv2d(c, embed_dim, kernel_size=1, bias=False) for c in feature_channels
        ])

        self.scale_embed = nn.Parameter(torch.zeros(self.num_scales, 1, 1, embed_dim))

        total_tokens = sum(p * p for p in pool_sizes)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = torch.linspace(0, drop_path_rate, steps=depth).tolist() if depth > 0 else []
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                layer_scale_init_value=layer_scale_init_value,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.scale_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.trunc_normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

        self._model_config = {
            'architecture': 'fontnext',
            'backbone': backbone_name,
            'embed_dim': embed_dim,
            'depth': depth,
            'num_heads': num_heads,
            'mlp_ratio': mlp_ratio,
            'pool_sizes': pool_sizes,
            'pretrained_backbone': pretrained_backbone,
            'drop_rate': drop_rate,
            'attn_drop_rate': attn_drop_rate,
            'drop_path_rate': drop_path_rate,
            'layer_scale_init_value': layer_scale_init_value,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        tokens = []

        for i, f in enumerate(feats):
            f = self.pools[i](f)
            f = self.projs[i](f)
            f = f.flatten(2).transpose(1, 2)
            f = f + self.scale_embed[i]
            tokens.append(f)

        x = torch.cat(tokens, dim=1)
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x[:, 0])
