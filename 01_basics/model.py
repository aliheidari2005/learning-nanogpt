import torch
import torch.nn as nn

from attention import CausalSelfAttention


# --------------------------------------------------
# Language Model
# --------------------------------------------------

class LanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        n_embd,
        block_size,
        num_heads
    ):

        super().__init__()

        # --------------------------------------------------
        # 1. Token Embedding
        # --------------------------------------------------

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=n_embd
        )

        # --------------------------------------------------
        # 2. Position Embedding
        # --------------------------------------------------

        self.position_embedding = nn.Embedding(
            num_embeddings=block_size,
            embedding_dim=n_embd
        )

        self.ln_1 = nn.LayerNorm(n_embd)

        # --------------------------------------------------
        # 3. Multi-Head Causal Self-Attention
        # --------------------------------------------------

        self.attention = CausalSelfAttention(
            n_embd=n_embd,
            block_size=block_size,
            num_heads=num_heads
        )

        # --------------------------------------------------
        # 4. Language Modeling Head
        # --------------------------------------------------

        self.lm_head = nn.Linear(
            in_features=n_embd,
            out_features=vocab_size
        )

    def forward(self, x):

        # x: [B, T]

        B, T = x.shape

        # --------------------------------------------------
        # Step 1: Token Embedding
        # --------------------------------------------------

        token_emb = self.token_embedding(x)

        # [B, T] -> [B, T, C]

        # --------------------------------------------------
        # Step 2: Position Embedding
        # --------------------------------------------------

        position_ids = torch.arange(
            T,
            device=x.device
        )

        pos_emb = self.position_embedding(position_ids)

        # [T] -> [T, C]

        # Broadcasting:
        # [B, T, C] + [T, C] -> [B, T, C]

        emb = token_emb + pos_emb


        # LayerNorm before Attention (Pre-LN)
        attn_input = self.ln_1(emb)


        # --------------------------------------------------
        # Step 3: Multi-Head Self-Attention
        # --------------------------------------------------

        attn_out = self.attention(attn_input)

        # Residual Connection
        x = emb + attn_out


        # [B, T, C] -> [B, T, C]

        # --------------------------------------------------
        # Step 4: Language Modeling Head
        # --------------------------------------------------

        logits = self.lm_head(x)

        # [B, T, C] -> [B, T, vocab_size]

        return logits