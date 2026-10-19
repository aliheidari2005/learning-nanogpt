import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, block_size, num_heads):
        super().__init__()

        assert n_embd % num_heads == 0, (
            "n_embd must be divisible by num_heads"
        )

        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_dim = n_embd // num_heads

        # Combined Query, Key, Value projection
        self.qkv = nn.Linear(n_embd, 3 * n_embd)

        # Output projection
        self.proj = nn.Linear(n_embd, n_embd)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(block_size, block_size)
            )
        )

    def forward(self, x, return_attention=False, return_head_outputs=False):
        B, T, C = x.shape

        # 1. Generate Q, K, V
        q, k, v = self.qkv(x).split(
            self.n_embd,
            dim=-1
        )

        # 2. Split into multiple heads
        q = q.view(
            B, T, self.num_heads, self.head_dim
        )

        k = k.view(
            B, T, self.num_heads, self.head_dim
        )

        v = v.view(
            B, T, self.num_heads, self.head_dim
        )

        # 3. Move head dimension before sequence dimension
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 4. Compute attention scores
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        # 5. Apply causal mask
        mask = self.causal_mask[:T, :T]

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # 6. Normalize attention scores
        attention_weights = F.softmax(
            scores,
            dim=-1
        )
        
        # 7. Aggregate values
        out = attention_weights @ v

        # Keep the output of each head before concatenation
        head_outputs = out

        # 8. Concatenate heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, self.n_embd)

        # 9. Output projection
        out = self.proj(out)

        if return_head_outputs:
            return out, attention_weights, head_outputs

        if return_attention:
            return out, attention_weights

        return out