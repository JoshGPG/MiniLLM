"""Tokenizer utility for MiniLLM.

This module provides a minimal whitespace tokenizer with a small
vocabulary builder. It supports adding special tokens required for the
MiniLLM experiments and exposes ``fit``, ``encode`` and ``decode``
methods.
"""

from collections import Counter
from typing import Iterable, List


class Tokenizer:
    """Whitespace tokenizer with a simple frequency-based vocabulary.

    The tokenizer builds a vocabulary from provided texts and always
    reserves IDs ``0-3`` for the special tokens ``<PAD>``, ``<BOS>``,
    ``<EOS>`` and ``<UNK>`` respectively.
    """

    pad_token: str = "<PAD>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    unk_token: str = "<UNK>"

    def __init__(self) -> None:
        self.special_tokens = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
        ]

        # Fixed IDs for the special tokens
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = range(4)

        # Vocabulary mappings
        self.token_to_id = {}
        self.id_to_token = {}

    # ------------------------------------------------------------------
    def fit(self, texts: Iterable[str], vocab_size: int = 8000) -> None:
        """Build vocabulary from ``texts``.

        The vocabulary is built by simple whitespace tokenisation. Tokens
        are counted and the most frequent ones are kept until the desired
        ``vocab_size`` (including special tokens) is reached.
        """

        # Reset vocab and initialise with special tokens
        self.token_to_id = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.special_tokens)}

        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(text.split())

        # Remove any occurrence of special tokens from the counts
        for tok in self.special_tokens:
            counter.pop(tok, None)

        limit = vocab_size - len(self.special_tokens)
        for token, _ in counter.most_common(limit):
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    # ------------------------------------------------------------------
    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> List[int]:
        """Convert ``text`` into a list of token IDs.

        Unknown tokens are mapped to ``<UNK>``. ``<BOS>`` and ``<EOS>``
        tokens can be optionally added to the beginning and end of the
        sequence.
        """

        ids = [
            self.token_to_id.get(tok, self.unk_id) for tok in text.split()
        ]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    # ------------------------------------------------------------------
    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Convert a list of IDs back into a string.

        Special tokens can be optionally removed from the returned text
        by setting ``skip_special_tokens`` to ``True``.
        """

        tokens: List[str] = []
        for idx in ids:
            token = self.id_to_token.get(idx, self.unk_token)
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return " ".join(tokens)


__all__ = ["Tokenizer"]

