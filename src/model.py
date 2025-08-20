"""Neural network model definition for MiniLLM."""

from typing import Tuple
import torch
from torch import nn

class MiniLLM(nn.Module):
    """A tiny language model built with PyTorch."""

    def __init__(self, vocab_size: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Forward pass that embeds token IDs and predicts logits."""
        x = self.embedding(ids)
        return self.linear(x)
