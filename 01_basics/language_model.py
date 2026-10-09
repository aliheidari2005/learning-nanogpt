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


print("\nBatch indices:")
print(ix)

print("\nInput x:")
print(x)

print("\nTarget y:")
print(y)

print("\nx.shape:", x.shape)
print("y.shape:", y.shape)

# --------------------------------------------------
# 3. Very simple language model
# --------------------------------------------------

vocab_size = len(vocab)
model = nn.Embedding(vocab_size, vocab_size)

# --------------------------------------------------
# 4. Forward
# --------------------------------------------------

logits = model(x)
print("\nLogits shape:", logits.shape)
print(logits)

# --------------------------------------------------
# 5. Loss
# --------------------------------------------------

loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    y.view(-1)
)
print("\nLoss:", loss.item())