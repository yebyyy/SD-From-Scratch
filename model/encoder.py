import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        # Each pixel is representing more features and the number of pixels is decreasing
        super().__init__(
            # (Batch_size, Channel, Height, Width) -> (Batch_size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),  # This won't change the size of the image

            # (Batch_size, 128, Height, Width) -> (Batch_size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            # (Batch_size, 128, Height, Width) -> (Batch_size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            # (Batch_size, 128, Height, Width) -> (Batch_size, 128, Height / 2, Width / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2), 

            # (Batch_size, 128, Height / 2, Width / 2) -> (Batch_size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(128, 256),  # Increasing the number of features

            # (Batch_size, 256, Height / 2, Width / 2) -> (Batch_size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256),

            # (Batch_size, 256, Height / 2, Width / 2) -> (Batch_size, 256, Height / 4, Width / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2), 

            # (Batch_size, 256, Height / 4, Width / 4) -> (Batch_size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(256, 512),
            
            # (Batch_size, 512, Height / 4, Width / 4) -> (Batch_size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height / 4, Width / 4) -> (Batch_size, 512, Height / 8, Width / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height / 8, Width / 8) -> (Batch_size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # self attention on each pixel, attention as a way to relate pixels to each other
            VAE_AttentionBlock(512),  # convolution is local, attention is global since the first pixel can relate to the last pixel

            VAE_ResidualBlock(512, 512),

            # (Group, Channel of Features)
            nn.GroupNorm(32, 512),

            nn.SiLU(),

            # (Batch_size, 512, Height / 8, Width / 8) -> (Batch_size, 8, Height / 8, Width / 8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (Batch_size, 8, Height / 8, Width / 8) -> (Batch_size, 8, Height / 8, Width / 8)
            nn.Conv2d(8, 8, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor):
        # x: (Batch_size, Channel, Height, Width)
        # noise: (Batch_size, Out_Channel, Height / 8, Width / 8)
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (Left, Right, Top, Bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # (Batch_size, 8, Height / 8, Width / 8) -> 2 * (Batch_size, 4, Height / 8, Width / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)  # Divide into 2 tensors in the channel dimension

        # clamp is used to prevent the log_variance from being too small or too large
        log_variance = torch.clamp(log_variance, -30, 20)

        # (Batch_size, 4, Height / 8, Width / 8) -> (Batch_size, 4, Height / 8, Width / 8)
        variance = log_variance.exp()

        # (Batch_size, 4, Height / 8, Width / 8) -> (Batch_size, 4, Height / 8, Width / 8)
        std = variance.sqrt()

        # Sample from the normal distribution
        x = mean + std * noise

        # Scale the mean and variance to the original size
        x *= 0.18215

        return x