import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    
    def __init__(self, vocab_size: int, n_embd: int, seq_length: int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(vocab_size, n_embd))  # Learned positional embedding

    def forward(self, tokens):
        # (batch_size, seq_length) -> (batch_size, seq_length, n_embd)
        x = self.token_embedding(tokens)

        x += self.position_embedding
        return x
    
class CLIPLayer(nn.Module):

    def __init__(self, n_heads: int, d_embd: int):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(d_embd)
        self.self_attention = SelfAttention(n_heads, d_embd)
        self.layernorm2 = nn.LayerNorm(d_embd)
        self.linear1 = nn.Linear(d_embd, 4 * d_embd)
        self.linear2 = nn.Linear(4 * d_embd, d_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_length, d_embd)
        residual = x

        x = self.layer_norm1(x)
        x = self.self_attention(x, causal_mask=True)
        x += residual

        residual = x
        x = self.layernorm2(x)
        x = self.linear1(x)
        x = x * torch.sigmoid(1.702 * x)  # QuickGELU
        x = self.linear2(x)
        x += residual

        return x
        



class CLIP(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)  # vocab_size, d_embd, seq_length

        self.layers = nn.Module([
            CLIPLayer(12, 768) for _ in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        
        # (batch_size, seq_length) -> (batch_size, seq_length, d_embd)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
        
        output = self.layernorm(state)
        return output