import torch
import torch.nn.functional as F

from data import build_vocabulary, encode, get_batch
from model import LanguageModel


# --------------------------------------------------
# 1. Tiny Dataset
# --------------------------------------------------

text = "hello world hello ali hello gpt"

words, vocab, stoi, itos = build_vocabulary(text)

data = encode(words, stoi)

print("Vocabulary:", vocab)
print("stoi:", stoi)
print("data:", data)


# --------------------------------------------------
# 2. Model Configuration
# --------------------------------------------------

n_embd = 8

num_heads = 2

batch_size = 2

block_size = 3

vocab_size = len(vocab)

learning_rate = 0.01

num_steps = 500


# --------------------------------------------------
# 3. Fixed Evaluation Batch
# --------------------------------------------------

eval_x, eval_y = get_batch(
    data,
    batch_size,
    block_size
)


# --------------------------------------------------
# 4. Initialize Model
# --------------------------------------------------

model = LanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    num_heads=num_heads
)


# --------------------------------------------------
# 5. Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# --------------------------------------------------
# 6. Save Initial Parameters
# --------------------------------------------------

initial_parameters = {
    name: parameter.detach().clone()
    for name, parameter in model.named_parameters()
}


# --------------------------------------------------
# 7. Initial Evaluation Loss
# --------------------------------------------------

model.eval()

with torch.no_grad():

    logits = model(eval_x)

    initial_eval_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        eval_y.reshape(-1)
    )


# --------------------------------------------------
# 8. Training Loop
# --------------------------------------------------

model.train()

for step in range(num_steps):

    # ----------------------------------------------
    # Get a fresh batch
    # ----------------------------------------------

    x, y = get_batch(
        data,
        batch_size,
        block_size
    )

    # ----------------------------------------------
    # Forward Pass
    # ----------------------------------------------

    logits = model(x)

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
# 9. Final Evaluation Loss
# --------------------------------------------------

model.eval()

with torch.no_grad():

    logits = model(eval_x)

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
# 10. Prediction
# --------------------------------------------------

with torch.no_grad():

    logits = model(x)

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
# 11. Compare Parameters After Training
# --------------------------------------------------

parameter_changes = {}

for name, parameter in model.named_parameters():

    change = (
        parameter.detach()
        - initial_parameters[name]
    ).abs().max().item()

    parameter_changes[name] = change


print("\nMaximum Parameter Changes:")


# Token Embedding

print(
    "Token Embedding:",
    parameter_changes["token_embedding.weight"]
)


# Position Embedding

print(
    "Position Embedding:",
    parameter_changes["position_embedding.weight"]
)


# Language Modeling Head

print(
    "LM Head Weights:",
    parameter_changes["lm_head.weight"]
)

print(
    "LM Head Bias:",
    parameter_changes["lm_head.bias"]
)


# Self-Attention

print("\nSelf-Attention Parameters:")

for name, change in parameter_changes.items():

    if name.startswith("attention."):

        print(f"{name.removeprefix('attention.')}: {change}")



# --------------------------------------------------
# 12. Inspect Multi-Head Attention Outputs
# --------------------------------------------------

model.eval()

with torch.no_grad():

    # Token embeddings
    token_emb = model.token_embedding(x)

    # Position embeddings
    position_ids = torch.arange(
        x.shape[1],
        device=x.device
    )

    pos_emb = model.position_embedding(position_ids)

    # Combined embeddings
    emb = token_emb + pos_emb

    # LayerNorm before Attention
    attn_input = model.ln_1(emb)

    # Get outputs and attention information
    attn_out, attention_weights, head_outputs = (
        model.attention(
            attn_input,
            return_head_outputs=True
        )
    )

    # Apply Residual Connection
    residual_out = emb + attn_out


print("\nAttention Weights Shape:")
print(attention_weights.shape)

print("\nHead Outputs Shape:")
print(head_outputs.shape)

sample_idx = 0

for head_idx in range(model.attention.num_heads):

    print(f"\nHead {head_idx + 1} Attention Weights:")

    print(
        attention_weights[
            sample_idx,
            head_idx
        ]
    )

    print(f"\nHead {head_idx + 1} Outputs:")

    print(
        head_outputs[
            sample_idx,
            head_idx
        ]
    )


print("\nConcatenated and Projected Attention Output:")

print(attn_out[sample_idx])