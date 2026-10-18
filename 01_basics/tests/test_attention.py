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

# --------------------------------------------------
# Test 4: Manual Attention vs PyTorch SDPA
# --------------------------------------------------

def test_attention_matches_scaled_dot_product_attention():

    torch.manual_seed(42)

    batch_size = 2
    block_size = 4
    n_embd = 8

    attention = CausalSelfAttention(
        n_embd=n_embd,
        block_size=block_size
    )

    attention.eval()

    x = torch.randn(
        batch_size,
        block_size,
        n_embd
    )

    with torch.no_grad():

        # Generate Q, K, V using the same learned parameters.
        Q = attention.query(x)
        K = attention.key(x)
        V = attention.value(x)

        # ------------------------------------------
        # Path 1: Manual attention
        # ------------------------------------------

        scores = Q @ K.transpose(-2, -1)
        scores = scores / (n_embd ** 0.5)

        mask = attention.causal_mask[
            :block_size,
            :block_size
        ]

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        weights = torch.softmax(
            scores,
            dim=-1
        )

        manual_out = weights @ V

        manual_out = attention.proj(manual_out)

        # ------------------------------------------
        # Path 2: PyTorch SDPA
        # ------------------------------------------

        sdpa_out = torch.nn.functional.scaled_dot_product_attention(
            Q,
            K,
            V,
            is_causal=True,
            dropout_p=0.0
        )

        sdpa_out = attention.proj(sdpa_out)

    # Both implementations must preserve the expected shape.
    assert manual_out.shape == (
        batch_size,
        block_size,
        n_embd
    )

    assert sdpa_out.shape == manual_out.shape

    # The numerical outputs should be close.
    torch.testing.assert_close(
        manual_out,
        sdpa_out,
        rtol=1e-5,
        atol=1e-6
    )

# --------------------------------------------------
# Test 5: Gradient Equivalence
# --------------------------------------------------

def test_attention_gradients_match_scaled_dot_product_attention():

    torch.manual_seed(42)

    batch_size = 2
    block_size = 4
    n_embd = 8

    # --------------------------------------------------
    # Create independent inputs for both implementations
    # --------------------------------------------------

    Q_manual = torch.randn(
        batch_size,
        block_size,
        n_embd,
        requires_grad=True
    )

    K_manual = torch.randn(
        batch_size,
        block_size,
        n_embd,
        requires_grad=True
    )

    V_manual = torch.randn(
        batch_size,
        block_size,
        n_embd,
        requires_grad=True
    )

    Q_sdpa = Q_manual.detach().clone().requires_grad_(True)
    K_sdpa = K_manual.detach().clone().requires_grad_(True)
    V_sdpa = V_manual.detach().clone().requires_grad_(True)

    # --------------------------------------------------
    # Path 1: Manual causal attention
    # --------------------------------------------------

    scores = Q_manual @ K_manual.transpose(-2, -1)

    scores = scores / (n_embd ** 0.5)

    causal_mask = torch.tril(
        torch.ones(block_size, block_size, dtype=torch.bool)
    )

    scores = scores.masked_fill(
        ~causal_mask,
        float("-inf")
    )

    weights = torch.softmax(scores, dim=-1)

    manual_out = weights @ V_manual

    manual_loss = manual_out.square().mean()

    manual_loss.backward()

    # Save gradients before running the second backward pass.
    manual_grads = (
        Q_manual.grad.detach().clone(),
        K_manual.grad.detach().clone(),
        V_manual.grad.detach().clone()
    )

    # --------------------------------------------------
    # Path 2: PyTorch SDPA
    # --------------------------------------------------

    sdpa_out = torch.nn.functional.scaled_dot_product_attention(
        Q_sdpa,
        K_sdpa,
        V_sdpa,
        is_causal=True,
        dropout_p=0.0
    )

    sdpa_loss = sdpa_out.square().mean()

    sdpa_loss.backward()

    sdpa_grads = (
        Q_sdpa.grad,
        K_sdpa.grad,
        V_sdpa.grad
    )

    # --------------------------------------------------
    # Compare outputs
    # --------------------------------------------------

    torch.testing.assert_close(
        manual_out,
        sdpa_out,
        rtol=1e-5,
        atol=1e-6
    )

    # --------------------------------------------------
    # Compare gradients
    # --------------------------------------------------

    for name, manual_grad, sdpa_grad in zip(
        ("Q", "K", "V"),
        manual_grads,
        sdpa_grads
    ):

        assert manual_grad is not None, (
            f"Manual gradient for {name} is missing"
        )

        assert sdpa_grad is not None, (
            f"SDPA gradient for {name} is missing"
        )

        assert torch.isfinite(manual_grad).all(), (
            f"Manual gradient for {name} contains NaN or Inf"
        )

        assert torch.isfinite(sdpa_grad).all(), (
            f"SDPA gradient for {name} contains NaN or Inf"
        )

        torch.testing.assert_close(
            manual_grad,
            sdpa_grad,
            rtol=1e-4,
            atol=1e-6,
            msg=f"Gradient mismatch for {name}"
        )

# --------------------------------------------------
# Test 6: Attention Weights and Causal Mask
# --------------------------------------------------

def test_attention_weights_respect_causal_mask():

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
        n_embd
    )

    # Request both output and attention weights
    with torch.no_grad():
        out, weights = attention(
            x,
            return_attention=True
        )

    # 1. Check attention weights shape
    assert weights.shape == (
        batch_size,
        block_size,
        block_size
    )

    # 2. Check output shape
    assert out.shape == (
        batch_size,
        block_size,
        n_embd
    )

    # 3. Attention weights must be finite
    assert torch.isfinite(weights).all()

    # 4. Each row must sum to 1
    row_sums = weights.sum(dim=-1)

    torch.testing.assert_close(
        row_sums,
        torch.ones_like(row_sums),
        rtol=1e-5,
        atol=1e-6
    )

    # 5. Future-token attention weights must be zero
    future_mask = torch.triu(
        torch.ones(
            block_size,
            block_size,
            dtype=torch.bool
        ),
        diagonal=1
    )

    future_weights = weights.masked_select(
        future_mask.unsqueeze(0)
    )

    torch.testing.assert_close(
        future_weights,
        torch.zeros_like(future_weights),
        rtol=0,
        atol=0
    )