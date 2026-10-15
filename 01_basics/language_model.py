import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --------------------------------------------------
# 1. Batch Generation
# --------------------------------------------------

def get_batch(data, batch_size, block_size):

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x, y


# --------------------------------------------------
# 2. Causal Self-Attention
# --------------------------------------------------

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


# --------------------------------------------------
# 3. Tiny Dataset
# --------------------------------------------------

text = "hello world hello ali hello gpt"

words = text.split()

vocab = sorted(set(words))

stoi = {
    word: i
    for i, word in enumerate(vocab)
}

itos = {
    i: word
    for word, i in stoi.items()
}

data = torch.tensor(
    [stoi[word] for word in words],
    dtype=torch.long
)

print("Vocabulary:", vocab)
print("stoi:", stoi)
print("data:", data)


# --------------------------------------------------
# 4. Model Configuration
# --------------------------------------------------

n_embd = 8
batch_size = 2
block_size = 3
vocab_size = len(vocab)


# --------------------------------------------------
# 5. Fixed Evaluation Batch
# --------------------------------------------------

eval_x, eval_y = get_batch(
    data,
    batch_size,
    block_size
)


# --------------------------------------------------
# 6. Model Components
# --------------------------------------------------

# 6.1 Token Embedding

token_embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=n_embd
)


# 6.2 Position Embedding

position_embedding = nn.Embedding(
    num_embeddings=block_size,
    embedding_dim=n_embd
)


# 6.3 Causal Self-Attention

attention = CausalSelfAttention(
    n_embd=n_embd,
    block_size=block_size
)


# 6.4 Language Modeling Head

lm_head = nn.Linear(
    in_features=n_embd,
    out_features=vocab_size
)


# --------------------------------------------------
# 7. Optimizer
# --------------------------------------------------

learning_rate = 0.01

all_parameters = (
    list(token_embedding.parameters())
    + list(position_embedding.parameters())
    + list(attention.parameters())
    + list(lm_head.parameters())
)

optimizer = torch.optim.AdamW(
    all_parameters,
    lr=learning_rate
)


# --------------------------------------------------
# 8. Save Initial Parameters
# --------------------------------------------------

initial_embedding_weights = (
    token_embedding.weight.detach().clone()
)

initial_position_weights = (
    position_embedding.weight.detach().clone()
)

initial_head_weights = (
    lm_head.weight.detach().clone()
)

initial_head_bias = (
    lm_head.bias.detach().clone()
)

initial_attention_parameters = {
    name: parameter.detach().clone()
    for name, parameter in attention.named_parameters()
}


# --------------------------------------------------
# 9. Position IDs
# --------------------------------------------------

position_ids = torch.arange(block_size)


# --------------------------------------------------
# 10. Training Configuration
# --------------------------------------------------

num_step = 500


# --------------------------------------------------
# 11. Initial Evaluation Loss
# --------------------------------------------------

with torch.no_grad():

    token_emb = token_embedding(eval_x)

    pos_emb = position_embedding(position_ids)

    emb = token_emb + pos_emb

    attn_out = attention(emb)

    logits = lm_head(attn_out)

    initial_eval_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        eval_y.reshape(-1)
    )


# --------------------------------------------------
# 12. Training Loop
# --------------------------------------------------

for step in range(num_step):

    # Get a fresh batch

    x, y = get_batch(
        data,
        batch_size,
        block_size
    )

    # ----------------------------------------------
    # Forward Pass
    # ----------------------------------------------

    token_emb = token_embedding(x)

    pos_emb = position_embedding(position_ids)

    emb = token_emb + pos_emb

    attn_out = attention(emb)

    logits = lm_head(attn_out)

    # ----------------------------------------------
    # Compute Loss
    # ----------------------------------------------

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1)
    )

    # ----------------------------------------------
    # Backpropagation
    # ----------------------------------------------

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    # ----------------------------------------------
    # Print Progress
    # ----------------------------------------------

    if (step + 1) % 50 == 0:

        print(
            f"Step {step + 1}: "
            f"Loss = {loss.item():.4f}"
        )


# --------------------------------------------------
# 13. Final Evaluation Loss
# --------------------------------------------------

with torch.no_grad():

    token_emb = token_embedding(eval_x)

    pos_emb = position_embedding(position_ids)

    emb = token_emb + pos_emb

    attn_out = attention(emb)

    logits = lm_head(attn_out)

    final_eval_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        eval_y.reshape(-1)
    )

print(
    f"\nInitial loss: "
    f"{initial_eval_loss.item():.4f}"
)

print(
    f"Final loss: "
    f"{final_eval_loss.item():.4f}"
)


# --------------------------------------------------
# 14. Prediction
# --------------------------------------------------

with torch.no_grad():

    token_emb = token_embedding(x)

    pos_emb = position_embedding(position_ids)

    emb = token_emb + pos_emb

    attn_out = attention(emb)

    logits = lm_head(attn_out)

    predictions = logits.argmax(dim=-1)


print("\nInput tokens:")
print(x)

print("\nTarget tokens:")
print(y)

print("\nPredictions:")
print(predictions)


print("\nDecoded predictions:")

for i in range(x.shape[0]):

    for j in range(x.shape[1]):

        input_word = itos[x[i, j].item()]

        target_word = itos[y[i, j].item()]

        predicted_word = itos[predictions[i, j].item()]

        print(
            f"Input: {input_word:5s} | "
            f"Target: {target_word:5s} | "
            f"Prediction: {predicted_word}"
        )


# --------------------------------------------------
# 15. Compare Parameters After Training
# --------------------------------------------------

embedding_change = (
    token_embedding.weight.detach()
    - initial_embedding_weights
).abs().max().item()


position_change = (
    position_embedding.weight.detach()
    - initial_position_weights
).abs().max().item()


head_weight_change = (
    lm_head.weight.detach()
    - initial_head_weights
).abs().max().item()


head_bias_change = (
    lm_head.bias.detach()
    - initial_head_bias
).abs().max().item()


attention_parameter_changes = {}

for name, parameter in attention.named_parameters():

    change = (
        parameter.detach()
        - initial_attention_parameters[name]
    ).abs().max().item()

    attention_parameter_changes[name] = change


print("\nMaximum Parameter Changes:")

print("Token Embedding:", embedding_change)

print("Position Embedding:", position_change)

print("LM Head Weights:", head_weight_change)

print("LM Head Bias:", head_bias_change)

print("\nSelf-Attention Parameters:")

for name, change in attention_parameter_changes.items():

    print(f"{name}: {change}")