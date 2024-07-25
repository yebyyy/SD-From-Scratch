import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()

        self.group_norm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, Channel, Height, Width)
        residual = x

        b, c, h, w = x.shape

        x = self.group_norm(x)

        # x: (Batch_size, Channel, Height * Width)
        x = x.view(b, c, h * w)

        # x: (Batch_size, Height * Width, Channel)
        x = x.transpose(-1, -2)  # Like each pixel having a feature(channel) vector

        # relate each pixel to each other
        x = self.attention(x)

        # x: (Batch_size, Channel, Height * Width)
        x = x.transpose(-1, -2)

        # x: (Batch_size, Channel, Height, Width)
        x = x.view(b, c, h, w)

        x += residual
        
        return x

class VAE_ResidualBlock(nn.Module):
    # This is made of convolutions and normalization

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.group_norm1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.group_norm2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual connection
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, In_Channels, Height, Width)

        residual = x

        x = self.group_norm1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.group_norm2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residual)