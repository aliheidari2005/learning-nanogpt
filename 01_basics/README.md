# 01 - Basics: A Tiny Language Model

This section implements a small language model from scratch
using PyTorch.

The goal is to understand the core components of a GPT-style
language model before building a full Transformer.

## Current Architecture

The model consists of:

1. Token Embedding
2. Learned Position Embedding
3. Single-Head Causal Self-Attention
4. Language Modeling Head

## Project Structure

- `data.py`: Vocabulary, encoding, and batch generation.
- `attention.py`: Manual implementation of causal self-attention.
- `model.py`: Language model architecture.
- `train.py`: Training, evaluation, prediction, and parameter analysis.
- `tests/`: Tests for attention behavior.
- `language_model.py`: Original monolithic implementation.

## Dataset

The current toy corpus is:

`hello world hello ali hello gpt`

The vocabulary contains four tokens:

- ali
- gpt
- hello
- world

## Tensor Shapes

For the current configuration:

- Batch size: 2
- Block size: 3
- Embedding dimension: 8
- Vocabulary size: 4

The main tensor shapes are:

| Component | Shape |
|---|---|
| Input tokens | `[B, T]` |
| Token embeddings | `[B, T, C]` |
| Position embeddings | `[T, C]` |
| Attention scores | `[B, T, T]` |
| Attention output | `[B, T, C]` |
| Logits | `[B, T, vocab_size]` |

## Training

Run from this directory:

```bash
python train.py