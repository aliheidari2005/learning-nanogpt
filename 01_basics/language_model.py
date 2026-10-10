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
# 4. Initial forward pass and loss
# --------------------------------------------------

# logits = model(x)

# print("\nLogits shape:", logits.shape)

# print("\nLogits:")
# print(logits)

# # logits: [B, T, V] -> [B*T, V]
# # y:      [B, T]    -> [B*T]

# loss = F.cross_entropy(
#     logits.reshape(-1, vocab_size),
#     y.reshape(-1)
# )

# initial_loss = loss.item()

# print("\nInitial loss:", initial_loss)


# # --------------------------------------------------
# # 5. Training setup
# # --------------------------------------------------

# learning_rate = 0.01
# num_steps = 500

# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=learning_rate
# )


# # --------------------------------------------------
# # 6. Training loop
# # --------------------------------------------------

# print("\nStarting training...\n")

# for step in range(num_steps):

#     # A. Forward pass
#     logits = model(x)

#     # B. Compute loss
#     loss = F.cross_entropy(
#         logits.reshape(-1, vocab_size),
#         y.reshape(-1)
#     )

#     # C. Clear gradients from the previous step
#     optimizer.zero_grad()

#     # D. Compute gradients
#     loss.backward()

#     # E. Update model parameters
#     optimizer.step()

#     # F. Print progress
#     if (step + 1) % 50 == 0:
#         print(
#             f"Step [{step + 1:3d}/{num_steps}] "
#             f"Loss: {loss.item():.6f}"
#         )


# # --------------------------------------------------
# # 7. Evaluate the trained model
# # --------------------------------------------------

# with torch.no_grad():

#     final_logits = model(x)

#     final_loss = F.cross_entropy(
#         final_logits.reshape(-1, vocab_size),
#         y.reshape(-1)
#     ).item()

# print("\nInitial loss:", initial_loss)
# print("Final loss:  ", final_loss)


# # --------------------------------------------------
# # 8. Inspect gradients and parameters
# # --------------------------------------------------

# print("\nModel parameters:")

# for name, param in model.named_parameters():

#     print(f"\nParameter: {name}")
#     print("Shape:", param.shape)
#     print("Requires grad:", param.requires_grad)

#     # Gradients remain available after optimizer.step()
#     if param.grad is not None:
#         print("Gradient shape:", param.grad.shape)
#         print("Gradient norm:", param.grad.norm().item())


# # --------------------------------------------------
# # 9. Inspect predictions
# # --------------------------------------------------

# with torch.no_grad():

#     logits = model(x)

#     # Predict the most likely next token
#     predictions = logits.argmax(dim=-1)

# print("\nInput tokens:")
# print(x)

# print("\nTarget tokens:")
# print(y)

# print("\nPredicted tokens:")
# print(predictions)

# print("\nDecoded predictions:")

# for b in range(batch_size):

#     print(f"\nSequence {b + 1}:")

#     for t in range(block_size):

#         input_word = itos[x[b, t].item()]
#         target_word = itos[y[b, t].item()]
#         predicted_word = itos[predictions[b, t].item()]

#         print(
#             f"Input: {input_word:5s} | "
#             f"Target: {target_word:5s} | "
#             f"Prediction: {predicted_word:5s}"
#         )