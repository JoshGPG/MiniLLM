"""Tokenizer utility for MiniLLM.

This module provides a minimal whitespace tokenizer with a small
vocabulary builder. It supports adding special tokens required for the
MiniLLM experiments and exposes ``fit``, ``encode`` and ``decode``
methods.

The vocabulary can be saved and restored using ``save_vocab`` and
``load_vocab``::

    tok = Tokenizer()
    tok.fit(texts)
    tok.save_vocab("vocab.json")

    new_tok = Tokenizer()
    new_tok.load_vocab("vocab.json")
"""

from collections import Counter
import json
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

    # ------------------------------------------------------------------
    def save_vocab(self, path: str) -> None:
        """Save the vocabulary to ``path`` in JSON format.

        The file stores the ``token_to_id`` mapping and IDs for the special
        tokens so that a subsequent load yields identical mappings.
        """

        data = {
            "token_to_id": self.token_to_id,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    def load_vocab(self, path: str) -> None:
        """Load a vocabulary from ``path``.

        The file must have been created by :meth:`save_vocab`. Both the
        token mappings and special token IDs are restored.
        """

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.token_to_id = {
            tok: int(idx) for tok, idx in data["token_to_id"].items()
        }
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}

        self.pad_id = data.get("pad_id", self.token_to_id.get(self.pad_token, 0))
        self.bos_id = data.get("bos_id", self.token_to_id.get(self.bos_token, 1))
        self.eos_id = data.get("eos_id", self.token_to_id.get(self.eos_token, 2))
        self.unk_id = data.get("unk_id", self.token_to_id.get(self.unk_token, 3))


__all__ = ["Tokenizer"]

