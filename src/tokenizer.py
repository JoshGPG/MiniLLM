"""Simple tokenizer utility for MiniLLM.

This tokenizer performs basic whitespace tokenization and mapping
between tokens and integer IDs. It is intentionally minimal and
serves as a starting point for experimentation.
"""

from typing import List

class SimpleTokenizer:
    """Tokenizes text by whitespace and maintains a vocabulary."""

    def __init__(self) -> None:
        self.token_to_id = {}
        self.id_to_token = {}

    def build_vocab(self, texts: List[str]) -> None:
        """Build a vocabulary from an iterable of texts."""
        for text in texts:
            for token in text.split():
                if token not in self.token_to_id:
                    idx = len(self.token_to_id)
                    self.token_to_id[token] = idx
                    self.id_to_token[idx] = token

    def encode(self, text: str) -> List[int]:
        """Convert a string into a list of token IDs."""
        return [self.token_to_id.get(tok, -1) for tok in text.split()]

    def decode(self, ids: List[int]) -> str:
        """Convert a list of token IDs back into a string."""
        return " ".join(self.id_to_token.get(i, "<unk>") for i in ids)
