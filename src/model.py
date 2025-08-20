"""Neural network model definition for MiniLLM."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class Embedding(nn.Module):
    """Lightweight wrapper around :class:`nn.Embedding`."""

    def __init__(self, vocab_size: int, emb_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(ids)


class PositionalEncoding(nn.Module):
    """Add positional information to token embeddings.

    Supports either sinusoidal positional encodings or learnable position
    embeddings via :class:`nn.Parameter`.
    """

    def __init__(
        self, emb_dim: int, max_seq_len: int, learnable: bool = False
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len
        self.learnable = learnable

        if learnable:
            # ``nn.Parameter`` so positions are learned during training.
            self.pos_embedding = nn.Parameter(
                torch.zeros(1, max_seq_len, emb_dim)
            )
        else:
            # Pre-compute sinusoidal embeddings and store as a buffer so it is
            # moved correctly across devices but not updated during training.
            position = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, emb_dim, 2).float()
                * (-math.log(10000.0) / emb_dim)
            )
            pe = torch.zeros(max_seq_len, emb_dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pos_embedding", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to ``x``.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(batch_size, seq_len, emb_dim)``.
        """

        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
            )
        return x + self.pos_embedding[:, :seq_len, :]


@dataclass
class ModelConfig:
    """Configuration for :class:`MiniLLM`."""

    vocab_size: int
    emb_dim: int
    max_seq_len: int = 512
    learnable_pos: bool = False


class MiniLLM(nn.Module):
    """A tiny language model built with PyTorch."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = Embedding(config.vocab_size, config.emb_dim)
        self.pos_encoding = PositionalEncoding(
            config.emb_dim, config.max_seq_len, config.learnable_pos
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.emb_dim, nhead=4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.linear = nn.Linear(config.emb_dim, config.vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Forward pass that embeds token IDs and predicts logits."""

        x = self.embedding(ids)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        return self.linear(x)

