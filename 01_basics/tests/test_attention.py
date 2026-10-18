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

# --------------------------------------------------
# Test 3: Gradient Flow
# --------------------------------------------------

def test_attention_gradient_flow():

    torch.manual_seed(42)

    batch_size = 2
    block_size = 4
    n_embd = 8

    attention = CausalSelfAttention(
        n_embd=n_embd,
        block_size=block_size
    )

    x = torch.randn(
        batch_size,
        block_size,
        n_embd,
        requires_grad=True
    )

    # Forward pass
    out = attention(x)

    # A simple differentiable objective for testing.
    loss = out.square().mean()

    # Backward pass
    loss.backward()

    # 1. Input gradient must exist and be finite.
    assert x.grad is not None

    assert torch.isfinite(x.grad).all()

    # 2. Input gradient must not be entirely zero.
    assert x.grad.abs().sum().item() > 0

    # 3. Every trainable parameter must have a gradient.
    for name, param in attention.named_parameters():

        assert param.grad is not None, (
            f"Missing gradient for parameter: {name}"
        )

        assert torch.isfinite(param.grad).all(), (
            f"Non-finite gradient for parameter: {name}"
        )

    # 4. The overall parameter gradient must be nonzero.
    total_grad = sum(
        param.grad.abs().sum().item()
        for param in attention.parameters()
        if param.grad is not None
    )

    assert total_grad > 0