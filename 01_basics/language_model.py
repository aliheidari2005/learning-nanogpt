import torch
import torch.nn as nn
import torch.nn.functional as F

def get_batch(data, batch_size, block_size):

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    x =torch.stack([data[i:i + block_size]
                   for i in ix
    ])

    y = torch.stack([data[i + 1:i + block_size + 1]
                    for i in ix
    ])

    return x,y

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
# 2. Fix sample
# --------------------------------------------------
eval_x, eval_y = get_batch(
    data,
    batch_size=2,
    block_size=3
)

# --------------------------------------------------
# 3. Simple language model
# --------------------------------------------------

# Model configuration
vocab_size = len(vocab)

n_embd = 8

# 3.1 Token Embedding
token_embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=n_embd,
)

# 3.2 Language Modeling Head
lm_head = nn.Linear(
    in_features=n_embd,
    out_features=vocab_size,
)

# --------------------------------------------------
# 4. Optimizer
# --------------------------------------------------

learning_rate = 0.01

optimizer = torch.optim.AdamW(list(token_embedding.parameters()) + list(lm_head.parameters()), lr=learning_rate)


# --------------------------------------------------
# 5. Save initial parameters
# --------------------------------------------------

initial_embedding_weights = (
    token_embedding.weight.detach().clone()
)

initial_head_weights = (
    lm_head.weight.detach().clone()
)

initial_head_bias = (
    lm_head.bias.detach().clone()
)



# --------------------------------------------------
# 6. Training
# --------------------------------------------------

num_step = 500

# Initial loss
with torch.no_grad():
    emb = token_embedding(eval_x)
    logits = lm_head(emb)

    initial_eval_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        eval_y.reshape(-1),
    )



# Training loop
for step in range(num_step):

    #grt fresh batch
    x, y = get_batch(
        data,
        batch_size=2,
        block_size=3
    )
    
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
    emb = token_embedding(eval_x)
    logits = lm_head(emb)

    final_eval_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        eval_y.reshape(-1),
    )
print(f"\nInitial loss: {initial_eval_loss.item():.4f}")
print(f"Final loss: {final_eval_loss.item():.4f}")


# --------------------------------------------------
# 7. Prediction
# --------------------------------------------------


with torch.no_grad():
    emb = token_embedding(x)
    logits = lm_head(emb)

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
# 8. Compare parameters after training
# --------------------------------------------------

embedding_change = (
    token_embedding.weight.detach()
    - initial_embedding_weights
).abs().max().item()

head_weight_change = (
    lm_head.weight.detach()
    - initial_head_weights
).abs().max().item()

head_bias_change = (
    lm_head.bias.detach()
    - initial_head_bias
).abs().max().item()

print("\nMaximum parameter changes:")
print("Embedding:", embedding_change)
print("LM head weights:", head_weight_change)
print("LM head bias:", head_bias_change)