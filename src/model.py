"""Neural network model definition for a mini Transformer."""

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


class SelfAttention(nn.Module):
    """Multi-head self-attention layer."""

    def __init__(
        self, emb_dim: int, num_heads: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out, _ = self.attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.dropout(out)


@dataclass
class ModelConfig:
    """Configuration for :class:`MiniTransformer`."""

    vocab_size: int
    emb_dim: int
    num_heads: int
    max_seq_len: int = 512
    learnable_pos: bool = False
    dropout: float = 0.0
    ffn_dim: int | None = None
    num_layers: int = 2
    pre_norm: bool = True
    tie_weights: bool = False


class TransformerBlock(nn.Module):
    """Self-attention block followed by a feed-forward network."""

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        ffn_dim: int | None = None,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()
        self.pre_norm = pre_norm
        self.attention = SelfAttention(emb_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        ffn_dim = ffn_dim or emb_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.attention(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
        else:
            x = self.norm1(x + self.attention(x))
            x = self.norm2(x + self.ffn(x))
        return x


class MiniTransformer(nn.Module):
    """A tiny Transformer-based language model built with PyTorch."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = Embedding(config.vocab_size, config.emb_dim)
        self.pos_encoding = PositionalEncoding(
            config.emb_dim, config.max_seq_len, config.learnable_pos
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    config.emb_dim,
                    config.num_heads,
                    dropout=config.dropout,
                    ffn_dim=config.ffn_dim,
                    pre_norm=config.pre_norm,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        if config.tie_weights:
            # Share weights between embedding and output projection
            self.lm_head.weight = self.embedding.embedding.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Forward pass that embeds token IDs and predicts logits."""

        x = self.embedding(ids)
        x = self.pos_encoding(x)
        for block in self.layers:
            x = block(x)
        logits = self.lm_head(x)
        return logits


# Backwards compatibility for older imports
MiniLLM = MiniTransformer

