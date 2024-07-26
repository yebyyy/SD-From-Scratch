import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):

    def __init__(self, d_embd: int):
        super().__init__()

        self.linear1 = nn.Linear(d_embd, 4 * d_embd)
        self.linear2 = nn.Linear(4 * d_embd, 4 * d_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear2(x)
        # (1, 1280)
        return x
    
class SwitchSequential(nn.Sequential):
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                latent = layer(latent, context)
            elif isinstance(layer, UNET_ResidualBlock):
                latent = layer(latent, time)
            else:
                latent = layer(latent)
        return latent

class UNET(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([
            # (Batch_size, 4, height / 8, width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
                             
            SwitchSequantial(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40))
            
            # Decrease size of the image
            # (Batch_size, 4, height / 8, width / 8) -> (Batch_size, 320, height / 16, width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
                             
            SwitchSequential(UNET_residualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_residualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            # (batch_size, 640, height / 16, width / 16) -> (batch_size, 640, height / 32, width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequantial(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequantial(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # (batch_size, 1280, height / 32, width / 32) -> (batch_size, 1280, height / 64, width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1))
        
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

            # (batch_size, 1280, height / 64, width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280))
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280)
        )
        
        self.decoders = nn.ModuleList([
            # (batch_size, 2560, height / 64, width / 64) -> (batch_size, 1280, height / 64, width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),  # Double the size because of skip connection

            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UpSample(1280, 640)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
                             
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), UpSample(1280)),  # 1920 = 1280 + 640

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), UpSample(640)),

            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),

        ])

# Diffusion class is basically the Unet
class Diffusion(nn.Module):

    def __init__(self):
        super().__init__()

        self.time_embedding = TimeEmbedding(320)  # Convert the time step(number) to a vector
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (batch_size, 4, height / 8, width / 8) from encoder
        # context(prompt): (batch_size, seq_length, d_embd) from clip
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (batch_size, 4, height / 8, width / 8) -> (batch_size, 320, height / 8, width / 8)
        output = self.unet(latent, context, time)

        # (batch_size, 320, height / 8, width / 8) -> (batch_size, 4, height / 8, width / 8)
        output = self.final(output)

        return output