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
    
class VAE_Decoder(nn.Sequential):
    
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height / 8, Width / 8) -> (Batch_size, 512, Height / 4, Width / 4)
            nn.Upsample(scale_factor=2),  # replicate the pixels twice on each dimension

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height / 4, Width / 4) -> (Batch_size, 512, Height / 2, Width / 2)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # (Batch_size, 256, Height / 2, Width / 2) -> (Batch_size, 256, Height, Width)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),
            nn.SiLU(),

            # (Batch_size, 128, Height, Width) -> (Batch_size, 3, Height, Width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, 4, Height / 8, Width / 8)
        x /= 0.18215

        for module in self:
            x = module(x)
        
        # x: (Batch_size, 3, Height, Width)
        return x