"""Training script for MiniLLM."""

from __future__ import annotations
import argparse
from typing import List

import torch
from torch import nn, optim

from .model import MiniLLM
from .tokenizer import SimpleTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MiniLLM model")
    parser.add_argument("--epochs", type=int, default=1, help="number of training epochs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(["hello world"])

    model = MiniLLM(vocab_size=len(tokenizer.token_to_id))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    dummy_ids = torch.tensor([tokenizer.encode("hello world")])
    targets = dummy_ids.clone()

    for _ in range(args.epochs):
        optimizer.zero_grad()
        outputs = model(dummy_ids)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

    print("Training complete")


if __name__ == "__main__":
    main()
