"""Neural network model definition for a mini Transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn

__all__ = ["ModelConfig", "MiniLLM"]


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


class Embedding(nn.Module):
    """Lightweight wrapper around :class:`nn.Embedding`."""

    def __init__(self, vocab_size: int, emb_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs.

        Parameters
        ----------
        ids: torch.Tensor
            Integer tensor of shape ``(batch_size, seq_len)`` with values in
            ``[0, vocab_size)`` and dtype ``torch.long``.

        Returns
        -------
        torch.Tensor
            Embedded tensor of shape ``(batch_size, seq_len, emb_dim)``.
        """

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
    """Configuration for :class:`MiniTransformer`.

    Parameters
    ----------
    vocab_size:
        Size of the vocabulary.
    max_seq_len:
        Maximum sequence length the model can process.
    emb_dim:
        Dimensionality of the token embeddings. Defaults to ``128``.
    num_layers:
        Number of :class:`TransformerBlock` layers. Defaults to ``2``.
    num_heads:
        Number of attention heads. Defaults to ``2``.
    ffn_dim:
        Hidden dimensionality of the feed-forward network. Defaults to ``256``.
    learnable_pos:
        If ``True``, positional encodings are learned. Otherwise sinusoidal
        encodings are used. Defaults to ``False``.
    dropout:
        Dropout probability. Defaults to ``0.0``.
    pre_norm:
        Whether to apply pre-normalization. Defaults to ``True``.
    tie_weights:
        Share token embedding weights with the output projection. Defaults to
        ``False``.
    """

    vocab_size: int
    max_seq_len: int = 512
    emb_dim: int = 128
    num_layers: int = 2
    num_heads: int = 2
    ffn_dim: int = 256
    learnable_pos: bool = False
    dropout: float = 0.0
    pre_norm: bool = True
    tie_weights: bool = False

    def __post_init__(self) -> None:
        """Validate parameter ranges."""

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.emb_dim <= 0:
            raise ValueError("emb_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.emb_dim % self.num_heads != 0:
            raise ValueError("emb_dim must be divisible by num_heads")
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be in the range [0, 1]")


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
        """Convert token IDs to logits.

        Pipeline
        --------
        1. Input tokens → embeddings → positional encodings.
        2. Iterate through transformer blocks.
        3. Feed final hidden states to output head → logits.

        Parameters
        ----------
        ids: torch.Tensor
            Tensor of token IDs with shape ``(batch_size, seq_len)`` and dtype
            ``torch.long``. Each entry should be in the range
            ``[0, config.vocab_size)``.

        Returns
        -------
        torch.Tensor
            Logit tensor of shape ``(batch_size, seq_len, config.vocab_size)``.
        """

        x = self.embedding(ids)  # (batch_size, seq_len, emb_dim)
        x = self.pos_encoding(x)  # (batch_size, seq_len, emb_dim)
        for block in self.layers:
            x = block(x)  # (batch_size, seq_len, emb_dim)
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        return logits


# Backwards compatibility for older imports
MiniLLM = MiniTransformer