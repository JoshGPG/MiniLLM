"""Neural network model definition for MiniLLM."""

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn


@dataclass
class ModelConfig:
    """Configuration for :class:`MiniLLM`."""

    vocab_size: int
    hidden_size: int = 32
    max_seq_len: int = 512
    tie_weights: bool = False
    positional_encoding: Optional[str] = None  # ``"sinusoidal"`` or ``None``


class MiniLLM(nn.Module):
    """A tiny language model built with PyTorch."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        if config.positional_encoding == "sinusoidal":
            pe = self._build_sinusoidal_pe(config.max_seq_len, config.hidden_size)
            self.register_buffer("positional_encoding", pe)
        else:
            self.positional_encoding = None

        self.linear = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        if config.tie_weights:
            self.linear.weight = self.embedding.weight

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, dim: int) -> torch.Tensor:
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Forward pass that embeds token IDs and predicts logits."""
        x = self.embedding(ids)
        if self.positional_encoding is not None:
            x = x + self.positional_encoding[: x.size(1)]
        return self.linear(x)
