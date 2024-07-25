import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention

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