import sys
from pathlib import Path

import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from attention import CausalSelfAttention


torch.manual_seed(42)

batch_size = 1
block_size = 4
n_embd = 8

attention = CausalSelfAttention(
    n_embd=n_embd,
    block_size=block_size
)

x = torch.randn(
    batch_size,
    block_size,
    n_embd
)

out, weights = attention(
    x,
    return_attention=True
)

print("Input shape:", x.shape)
print("Output shape:", out.shape)
print("Attention weights shape:", weights.shape)

print("\nAttention weights:")
print(weights[0])

print("\nRow sums:")
print(weights[0].sum(dim=-1))

print("\nUpper triangular weights:")
print(torch.triu(weights[0], diagonal=1))