"""Neural network model definition for MiniLLM."""

import torch
from torch import nn


class Embedding(nn.Module):
    """Lightweight wrapper around :class:`nn.Embedding`."""

    def __init__(self, vocab_size: int, emb_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(ids)


class MiniLLM(nn.Module):
    """A tiny language model built with PyTorch."""

    def __init__(self, vocab_size: int, emb_dim: int) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size, emb_dim)
        self.linear = nn.Linear(emb_dim, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Forward pass that embeds token IDs and predicts logits."""
        x = self.embedding(ids)
        return self.linear(x)
