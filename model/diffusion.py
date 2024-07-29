import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):

    def __init__(self, d_embd: int):
        super().__init__()

        self.linear_1 = nn.Linear(d_embd, 4 * d_embd)
        self.linear_2 = nn.Linear(4 * d_embd, 4 * d_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        # (1, 1280)
        return x
    
class UNET_ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, d_time:int = 1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(d_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, latent, time):
        # latent: (batch_size, in_channels, height, width)
        # time: (1, 1280)
        residual = latent
        latent = self.groupnorm_feature(latent)
        latent = F.silu(latent)
        latent = self.conv_feature(latent)
        latent = self.conv_feature(latent)

        time = F.silu(time)
        time = self.linear_time(time)

        merged = latent + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residual)
        
class UNET_AttentionBlock(nn.Module):

    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        
        self.groupnorm = nn.GroupNorm(32, channels)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, latent, context):
        # latent: (batch_size, channels, height, width)
        # context: (batch_size, seq_length, d_context)

        long_residual = latent
        
        latent = self.groupnorm(latent)
        latent = self.conv_input(latent)

        b, c, h, w = latent.shape
        latent = latent.view(b, c, h * w)
        # (batch_size, height * width, channels)
        latent = latent.transpose(-1, -2)

        # Normalization + Self-Attention with residual connection
        short_residual = latent
        latent = self.layernorm_1(latent)
        latent = self.attention_1(latent)
        latent = latent + short_residual

        # Normalization + Cross-Attention with residual connection
        short_residual = latent
        latent = self.layernorm_2(latent)
        latent = self.attention_2(latent, context)
        latent = latent + short_residual

        # Normalization + Feedforward with residual connection
        short_residual = latent
        latent = self.layernorm_3(latent)
        latent, gate = self.linear_geglu_1(latent).chunk(2, dim=-1)  # Split the tensor into two parts
        latent = latent * F.gelu(gate)  # element-wise multiplication
        latent = self.linear_geglu_2(latent)
        latent = latent + short_residual

        # (batch_size, channels, height, width)
        latent = latent.transpose(-1, -2).view((b, c, h, w))

        latent = latent + long_residual

class UpSample(nn.Module):

    def __init__(self, channels: int):
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, Channel, Height, Width) -> (Batch_size, Channel, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # nearest: replicate the pixels, basically the same as nn.Upsample
        return self.conv(x)

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
                             
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # Decrease size of the image
            # (Batch_size, 4, height / 8, width / 8) -> (Batch_size, 320, height / 16, width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
                             
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            # (batch_size, 640, height / 16, width / 16) -> (batch_size, 640, height / 32, width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # (batch_size, 1280, height / 32, width / 32) -> (batch_size, 1280, height / 64, width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
        
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

            # (batch_size, 1280, height / 64, width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
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

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UpSample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
                             
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), UpSample(1280)),  # 1920 = 1280 + 640

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), UpSample(640)),

            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),

        ])

    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x

class UNET_OutputLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 320, height / 8, width / 8)
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        # (batch_size, 4, height / 8, width / 8)
        return x

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