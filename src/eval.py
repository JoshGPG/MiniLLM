"""Evaluation script for MiniLLM."""

from __future__ import annotations
import argparse
import torch

from .model import MiniLLM
from .tokenizer import SimpleTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the MiniLLM model")
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(["hello world"])
    model = MiniLLM(vocab_size=len(tokenizer.token_to_id))
    ids = torch.tensor([tokenizer.encode("hello world")])
    with torch.no_grad():
        logits = model(ids)
    print("Logits:", logits)


if __name__ == "__main__":
    main()
