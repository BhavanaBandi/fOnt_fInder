import torch
import torch.nn as nn

def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    """ResNet-style residual block (2×3×3 conv with skip connection)."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            # 1×1 convolution to match dimensions if needed
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class CNNStem(nn.Module):
    """Convolutional stem: initial conv layers to extract local features."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # -> (batch,64,112,112)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)             # -> (batch,64,56,56)
        # Residual layers (blocks): downsample to 14×14
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1)
        )  # stays at (56×56)
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),  # -> (batch,128,28,28)
            ResidualBlock(128, 128, stride=1)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2), # -> (batch,256,14,14)
            ResidualBlock(256, 256, stride=1)
        )
        # (Optionally, a layer4 could be added for 7x7 output.)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x  # (batch,256,14,14)

class FontClassifier(nn.Module):
    """
    Hybrid CNN-Transformer model for font classification (3,812 classes).
    """
    def __init__(self, num_classes=3812, transformer_layers=6, d_model=512, num_heads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.stem = CNNStem()
        # After CNN stem: (batch, 256, 14, 14)
        num_tokens = 14 * 14
        # Linear projection from CNN channels to transformer dim
        self.token_proj = nn.Linear(256, d_model)
        # Class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, d_model))
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        # Final normalization and classification head
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

        # (Optional) Initialize parameters if desired, e.g., cls_token and pos_embed.

    def forward(self, x):
        # CNN stem
        x = self.stem(x)  # (batch,256,14,14)
        b, c, h, w = x.size()
        # Flatten HxW to sequence of tokens
        x = x.flatten(2).transpose(1, 2)  # -> (batch, h*w, 256)
        x = self.token_proj(x)           # -> (batch, num_tokens, d_model)
        # Prepare class token
        cls_tokens = self.cls_token.expand(b, -1, -1)  # (batch,1,d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_tokens+1, d_model)
        # Add positional embeddings
        x = x + self.pos_embed
        # Transformer encoder
        x = self.transformer(x)  # (batch, num_tokens+1, d_model)
        # Use class token representation
        cls_rep = x[:, 0, :]     # (batch, d_model)
        # Classification head
        out = self.norm(cls_rep)
        out = self.dropout(out)
        logits = self.classifier(out)  # (batch, num_classes)
        return logits
