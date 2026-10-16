import torch


# --------------------------------------------------
# 1. Build Vocabulary
# --------------------------------------------------

def build_vocabulary(text):
    """
    Build vocabulary and token mappings from text.

    Returns:
        words: List of words in the corpus
        vocab: Sorted list of unique words
        stoi: Mapping from word to token ID
        itos: Mapping from token ID to word
    """

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

    return words, vocab, stoi, itos


# --------------------------------------------------
# 2. Encode Text
# --------------------------------------------------

def encode(words, stoi):
    """
    Convert words into integer token IDs.
    """

    return torch.tensor(
        [stoi[word] for word in words],
        dtype=torch.long
    )


# --------------------------------------------------
# 3. Generate Batch
# --------------------------------------------------

def get_batch(data, batch_size, block_size):
    """
    Generate input and target batches.

    x contains input token sequences.
    y contains the same sequences shifted by one token.
    """

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