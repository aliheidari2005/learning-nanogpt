# 01 - Basics: Building a Language Model from Scratch

## Overview

This section introduces the fundamental components of a GPT-style language model using PyTorch.

The goal is to understand how a language model processes tokens, represents their meanings and positions, computes causal self-attention, and learns to predict the next token.

All components are implemented incrementally to understand the underlying mathematics, tensor operations, and training process rather than copying an existing implementation.

## Learning Objectives

By completing this section, the goal is to understand:

- Vocabulary construction and token encoding
- Input and target sequence generation
- Token embeddings
- Learned positional embeddings
- Query, Key, and Value projections
- Scaled dot-product attention
- Causal masking
- Attention weights and weighted Value aggregation
- Language modeling logits
- Cross-entropy loss
- Backpropagation and parameter optimization
- Basic model evaluation and prediction
- Modular PyTorch project organization
- Unit testing for neural network components

---

## Project Structure

```text
01_basics/
├── README.md
├── language_model.py
├── data.py
├── attention.py
├── model.py
├── train.py
├── tests/
│   └── test_attention.py
└── notebooks/
```

### File Responsibilities

| File | Responsibility |
|---|---|
| `language_model.py` | Original monolithic implementation retained for reference |
| `data.py` | Vocabulary construction, token encoding, and batch generation |
| `attention.py` | Manual implementation of causal self-attention |
| `model.py` | Language model architecture and forward pass |
| `train.py` | Training, evaluation, prediction, and parameter analysis |
| `tests/test_attention.py` | Tests for attention output shape and causal behavior |
| `notebooks/` | Interactive experiments and visualizations |

---

## 1. Dataset and Vocabulary

The current model uses a small toy corpus:

```text
hello world hello ali hello gpt
```

The vocabulary is constructed from the unique words in the corpus.

Expected vocabulary:

```python
['ali', 'gpt', 'hello', 'world']
```

The word-to-index mapping is:

```python
{
    'ali': 0,
    'gpt': 1,
    'hello': 2,
    'world': 3
}
```

The encoded token sequence is:

```python
tensor([2, 3, 2, 0, 2, 1])
```

The vocabulary and encoding functions are implemented in `data.py`.

### Important Concepts

- Vocabulary
- Token IDs
- Encoding and decoding
- Integer representations of text
- Input sequences and next-token targets

---

## 2. Batch Generation

The `get_batch()` function generates input sequences and their corresponding next-token targets.

For example:

```text
Input:  hello world hello
Target: world hello ali
```

The target sequence is shifted one position to the right relative to the input sequence.

This teaches the model to predict the next token at each position.

For batch size `B` and sequence length `T`, the shapes are:

| Tensor | Shape |
|---|---|
| Input batch `x` | `[B, T]` |
| Target batch `y` | `[B, T]` |

Current configuration:

```python
batch_size = 2
block_size = 3
```

Therefore:

```text
x.shape = [2, 3]
y.shape = [2, 3]
```

---

## 3. Token Embeddings

Each token ID is mapped to a learnable vector.

The current configuration uses:

```python
n_embd = 8
```

With vocabulary size 4, the token embedding matrix has shape:

```text
[4, 8]
```

For an input batch of shape `[B, T]`, the resulting token embeddings have shape:

```text
[B, T, C]
```

where:

- `B` is the batch size.
- `T` is the sequence length.
- `C` is the embedding dimension.

For the current configuration:

```text
Input:            [2, 3]
Token embeddings: [2, 3, 8]
```

Token embeddings are learned during training through backpropagation.

---

## 4. Learned Positional Embeddings

Self-attention does not inherently encode the order of tokens.

To provide positional information, the model uses a learned positional embedding table.

The position IDs for a sequence of length 3 are:

```python
tensor([0, 1, 2])
```

The corresponding positional embeddings have shape:

```text
[T, C] = [3, 8]
```

They are added to the token embeddings:

```python
emb = token_emb + pos_emb
```

Using broadcasting:

```text
Token embeddings:      [2, 3, 8]
Positional embeddings:    [3, 8]
--------------------------------
Combined embeddings:   [2, 3, 8]
```

The resulting representation contains both token-specific and positional information.

---

## 5. Causal Self-Attention

The `CausalSelfAttention` class is implemented manually in `attention.py`.

It consists of four learned linear projections:

- Query projection
- Key projection
- Value projection
- Output projection

### 5.1 Query, Key, and Value

Given an input representation:

```text
X: [B, T, C]
```

the model computes:

```text
Q = Linear(X)
K = Linear(X)
V = Linear(X)
```

All three tensors have shape:

```text
[B, T, C]
```

Query and Key determine the compatibility between token positions.

Value contains the representations that are aggregated according to the attention weights.

### 5.2 Attention Scores

The attention scores are calculated using:

\[
S = \frac{QK^T}{\sqrt{C}}
\]

The tensor operations are:

```text
Q:        [B, T, C]
K^T:      [B, C, T]
-------------------
Scores:   [B, T, T]
```

Each score represents the compatibility between a query position and a key position.

The scaling factor helps control the magnitude of the scores before applying softmax.

### 5.3 Causal Masking

Autoregressive language models must not use future tokens when predicting the next token.

For a sequence of length 3, the causal mask is:

```text
[[1, 0, 0],
 [1, 1, 0],
 [1, 1, 1]]
```

Rows correspond to query positions, and columns correspond to key positions.

A value of `1` indicates an allowed attention connection. A value of `0` indicates a blocked connection.

Blocked scores are replaced with negative infinity:

```python
scores = scores.masked_fill(
    mask == 0,
    float("-inf")
)
```

After softmax, the blocked positions receive zero attention weight.

### 5.4 Attention Weights

The attention weights are calculated as:

\[
A = \operatorname{softmax}(S)
\]

Softmax is applied along the final dimension.

The resulting shape is:

```text
[B, T, T]
```

Each row sums to one, subject to numerical precision.

### 5.5 Weighted Value Aggregation

The output of the attention operation is:

\[
O = AV
\]

The tensor shapes are:

```text
Attention weights: [B, T, T]
Values:            [B, T, C]
--------------------------------
Output:            [B, T, C]
```

Each position receives a weighted combination of Value vectors from the positions it is allowed to attend to.

### 5.6 Output Projection

The aggregated representation is passed through a learned output projection.

```python
out = self.proj(out)
```

The output shape remains:

```text
[B, T, C]
```

The current implementation uses a single attention head.

---

## 6. Language Model Architecture

The `LanguageModel` class is implemented in `model.py`.

The current architecture is:

```text
Input Token IDs
      |
      v
Token Embedding
      |
      + <--- Position Embedding
      |
      v
Combined Embeddings
      |
      v
Causal Self-Attention
      |
      v
Language Modeling Head
      |
      v
Logits
```

The model consists of:

1. Token Embedding
2. Learned Position Embedding
3. Single-Head Causal Self-Attention
4. Language Modeling Head

The language modeling head maps the embedding dimension to the vocabulary size.

With the current configuration:

```text
Embedding dimension: 8
Vocabulary size:     4
```

the output shape is:

```text
[B, T, 4]
```

For example:

```text
Input shape:  [2, 3]
Logits shape: [2, 3, 4]
```

Each position produces four logits, one for each vocabulary token.

These logits represent unnormalized scores for the next-token prediction.

---

## 7. Training

Training is implemented in `train.py`.

The current configuration is:

```python
n_embd = 8
batch_size = 2
block_size = 3
learning_rate = 0.01
num_steps = 500
```

The optimizer is AdamW:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)
```

The training process consists of:

1. Sampling a batch.
2. Running the forward pass.
3. Computing cross-entropy loss.
4. Clearing previously accumulated gradients.
5. Computing gradients through backpropagation.
6. Updating model parameters.
7. Reporting the loss periodically.

The loss is computed as:

```python
loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    y.reshape(-1)
)
```

The logits are reshaped from:

```text
[B, T, vocab_size]
```

to:

```text
[B * T, vocab_size]
```

The targets are reshaped from:

```text
[B, T]
```

to:

```text
[B * T]
```

This allows cross-entropy to evaluate the next-token prediction at every position in the batch.

### Training Objective

The model learns to assign higher scores to the correct next token.

The cross-entropy objective for a target token \(y\) is:

\[
L = -\log p(y \mid \text{context})
\]

The model parameters are updated to reduce this loss.

---

## 8. Evaluation and Predictions

The training script evaluates the model on a fixed batch generated before training.

It reports:

- Initial evaluation loss
- Final evaluation loss
- Input token IDs
- Target token IDs
- Predicted token IDs
- Decoded input, target, and predicted words

Predictions are obtained using:

```python
predictions = logits.argmax(dim=-1)
```

This selects the token with the highest logit at each position.

The current evaluation batch is used to observe how the model changes during training.

**Important limitation:** The corpus is extremely small, and the evaluation batch comes from the same corpus used for training. Therefore, a low evaluation loss does not demonstrate generalization to unseen text.

A separate validation dataset and more meaningful evaluation procedures will be introduced in later stages.

---

## 9. Parameter Analysis

The training script records the initial parameter values and compares them with the final values after training.

It measures the maximum absolute change for each parameter tensor:

\[
\Delta_{\max} =
\max_i \left|\theta_i^{\text{final}} -
\theta_i^{\text{initial}}\right|
\]

The analysis includes:

- Token embedding weights
- Position embedding weights
- Language modeling head weights and bias
- Query projection parameters
- Key projection parameters
- Value projection parameters
- Attention output projection parameters

These measurements confirm that parameters have changed during training.

However, a larger parameter change does not necessarily mean that a parameter is more important to the model.

---

## 10. Testing

Basic attention tests are located in:

```text
tests/test_attention.py
```

The current tests check:

1. Whether the attention output has the expected shape.
2. Whether changing a future input position affects earlier output positions.

The second test verifies an important property of causal self-attention.

Run the tests from the repository root:

```bash
python -m pytest 01_basics/tests/test_attention.py -v
```

Run the training script from the repository root:

```bash
python 01_basics/train.py
```

---

## 11. Current Limitations

This implementation is an educational prototype rather than a complete GPT model.

Current limitations include:

- Very small toy corpus
- Word-level tokenization
- Single-head self-attention
- No residual connections
- No LayerNorm
- No feed-forward network
- No stack of Transformer blocks
- No autoregressive text generation loop
- No separate validation dataset
- No systematic comparison with a reference implementation

These limitations are intentional at this stage. Each component will be introduced and tested incrementally.

---

## 12. Learning Roadmap

The planned development sequence is:

### Stage 1: Validate Single-Head Attention

- Test tensor shapes.
- Test causal masking.
- Inspect attention weights.
- Verify gradient flow.
- Compare attention calculations with a PyTorch reference implementation.

### Stage 2: Multi-Head Attention

- Understand why multiple heads are useful.
- Implement head splitting and concatenation.
- Verify all tensor dimensions.
- Test causal behavior.

### Stage 3: Transformer Block

- Implement residual connections.
- Introduce LayerNorm.
- Implement the feed-forward network.
- Understand the order of operations in a Transformer block.

### Stage 4: Multiple Transformer Blocks

- Stack Transformer blocks.
- Introduce model configuration parameters.
- Inspect parameter counts and tensor shapes.

### Stage 5: GPT-Style Language Model

- Combine token embeddings and positional information.
- Stack Transformer blocks.
- Add the final normalization and language modeling head.
- Train the model end to end.

### Stage 6: Text Generation

- Implement autoregressive generation.
- Explore greedy decoding and sampling.
- Understand temperature and context length.

### Stage 7: Evaluation and Improvement

- Introduce separate training and validation data.
- Study overfitting and generalization.
- Experiment with learning rates and model dimensions.
- Compare the implementation with a reference GPT implementation.

---

## Learning Philosophy

The objective of this project is not simply to produce a working language model.

Each component should be understood through:

1. Intuition
2. Mathematical formulation
3. Tensor dimensions
4. Independent implementation
5. Tests and experiments
6. Analysis of results

The implementation will grow incrementally, with an emphasis on understanding the underlying concepts and writing the code independently.
