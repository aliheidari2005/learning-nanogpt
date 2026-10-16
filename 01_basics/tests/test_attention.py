import sys
from pathlib import Path

import torch

# Allow importing modules from the 01_basics directory.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from attention import CausalSelfAttention


# --------------------------------------------------
# Test 1: Output Shape
# --------------------------------------------------

def test_attention_output_shape():

    batch_size = 2
    block_size = 3
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

    out = attention(x)

    assert out.shape == (
        batch_size,
        block_size,
        n_embd
    )


# --------------------------------------------------
# Test 2: Causal Behavior
# --------------------------------------------------

def test_attention_does_not_use_future_tokens():

    torch.manual_seed(42)

    batch_size = 1
    block_size = 3
    n_embd = 8

    attention = CausalSelfAttention(
        n_embd=n_embd,
        block_size=block_size
    )

    attention.eval()

    x1 = torch.randn(
        batch_size,
        block_size,
        n_embd
    )

    x2 = x1.clone()

    # Change only the final token.
    x2[:, 2, :] += 100.0

    with torch.no_grad():

        out1 = attention(x1)

        out2 = attention(x2)

    # The first two positions must not change.
    assert torch.allclose(
        out1[:, :2, :],
        out2[:, :2, :],
        atol=1e-6
    )