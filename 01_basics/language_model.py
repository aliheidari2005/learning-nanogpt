import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# 1. Tiny dataset
# --------------------------------------------------

text = "hello world hello ali hello gpt"

words = text.split()
vocab = sorted(set(words))

stoi = {word: i for i, word in enumerate(vocab)}
itos = {i: word for word, i in stoi.items()}

data = torch.tensor(
    [stoi[word] for word in words],
    dtype=torch.long
)

print("Vocabulary:", vocab)
print("stoi:", stoi)
print("data:", data)


# --------------------------------------------------
# 2. Training samples
# --------------------------------------------------

batch_size = 2
block_size = 3

# Select random starting positions
ix = torch.randint(
    len(data) - block_size,
    (batch_size,)
)

# Input sequences
x = torch.stack([
    data[i:i + block_size]
    for i in ix
])

# Next-token targets
y = torch.stack([
    data[i + 1:i + block_size + 1]
    for i in ix
])

print("\nBatch indices:")
print(ix)

print("\nInput x:")
print(x)

print("\nTarget y:")
print(y)

print("\nx.shape:", x.shape)
print("y.shape:", y.shape)


# --------------------------------------------------
# 3. Simple language model
# --------------------------------------------------

# Model configuration
vocab_size = len(vocab)
n_embd = 8

# 1. Token Embedding
token_embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=n_embd,
)

# 2. Language Modeling Head
lm_head = nn.Linear(
    in_features=n_embd,
    out_features=vocab_size,
)

# Forward pass
emb = token_embedding(x)
logits = lm_head(emb)

# Compute loss
loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    y.reshape(-1),
)

print("Embedding shape:", emb.shape)
print("Logits shape:", logits.shape)
print("Loss:", loss.item())



print("Embedding parameters:", sum(
    p.numel() for p in token_embedding.parameters()
))

print("LM head parameters:", sum(
    p.numel() for p in lm_head.parameters()
))

# --------------------------------------------------
# 4. Optimizer
# --------------------------------------------------

learning_rate = 0.01

optimizer = torch.optim.AdamW(list(token_embedding.parameters()) + list(lm_head.parameters()), lr=learning_rate)


# --------------------------------------------------
# 5. Training
# --------------------------------------------------

num_step = 500

# Initial loss
with torch.no_grad():
    emb = token_embedding(x)
    logits = lm_head(emb)

    initial_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1),
    )

print(f"\nInitial loss: {initial_loss.item():.4f}")



# Training loop
for step in range(num_step):
    
    # Forward pass
    emb = token_embedding(x)
    logits = lm_head(emb)

    # Compute loss
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1),)

    # Clear previous gradients
    optimizer.zero_grad()

    # Compute gradients
    loss.backward()

    # Update parameters
    optimizer.step()

    # Print progress
    if (step + 1) % 50 == 0:
        print(f"Step {step + 1}: Loss = {loss.item():.4f}")

# Final loss
with torch.no_grad():
    emb = token_embedding(x)
    logits = lm_head(emb)

    final_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1),
    )

print(f"\nFinal loss: {final_loss.item():.4f}")