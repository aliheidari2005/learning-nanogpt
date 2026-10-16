import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, block_size):

        super().__init__()

        # Learnable projections: X -> Q, K, V
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # Output projection
        self.proj = nn.Linear(n_embd, n_embd)

        # Causal mask: lower triangular matrix
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.n_embd = n_embd

    def forward(self, x):
        
        # x: [B, T, C]
        B, T, C = x.shape

        # ------------------------------------------
        # Step 1: Generate Q, K, V
        # ------------------------------------------

        Q = self.query(x)  # [B, T, C]
        K = self.key(x)    # [B, T, C]
        V = self.value(x)  # [B, T, C]

        # ------------------------------------------
        # Step 2: Calculate attention scores
        # ------------------------------------------

        scores = Q @ K.transpose(-2, -1)

        # [B, T, C] @ [B, C, T] -> [B, T, T]

        scores = scores / math.sqrt(C)

        # ------------------------------------------
        # Step 3: Apply causal mask
        # ------------------------------------------

        mask = self.causal_mask[:T, :T]

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # ------------------------------------------
        # Step 4: Convert scores into attention weights
        # ------------------------------------------

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # ------------------------------------------
        # Step 5: Weighted sum of values
        # ------------------------------------------

        out = attention_weights @ V

        # [B, T, T] @ [B, T, C] -> [B, T, C]

        # ------------------------------------------
        # Step 6: Output projection
        # ------------------------------------------

        out = self.proj(out)

        return out

