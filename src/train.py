"""Training script for MiniLLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from torch import nn, optim

from .model import MiniLLM
from .tokenizer import Tokenizer


VOCAB_PATH = Path("data/vocab.json")
TRAIN_PATH = Path("data/splits/train.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MiniLLM model")
    parser.add_argument("--epochs", type=int, default=1, help="number of training epochs")
    return parser.parse_args()


def load_training_texts(path: Path) -> List[str]:
    """Load training texts from a JSON or JSONL file.

    The expected format is a sequence of dictionaries each containing
    ``question`` and ``answer`` keys. The two fields are concatenated to
    form the final training text.
    """

    if not path.exists():
        return ["hello world"]

    texts: List[str] = []
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            q = item.get("question", "")
            a = item.get("answer", "")
            texts.append(f"{q} {a}".strip())
    elif path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                q = item.get("question", "")
                a = item.get("answer", "")
                texts.append(f"{q} {a}".strip())
    else:
        raise ValueError(f"Unsupported training file format: {path.suffix}")
    return texts


def main() -> None:
    args = parse_args()

    train_texts = load_training_texts(TRAIN_PATH)

    tokenizer = Tokenizer()
    if VOCAB_PATH.exists():
        tokenizer.load_vocab(str(VOCAB_PATH))
    else:
        tokenizer.fit(train_texts)
        VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save_vocab(str(VOCAB_PATH))

    encoded = [tokenizer.encode(t, add_bos=True, add_eos=True) for t in train_texts]

    max_len = max(len(seq) for seq in encoded)
    inputs = torch.full((len(encoded), max_len), tokenizer.pad_id, dtype=torch.long)
    for i, seq in enumerate(encoded):
        inputs[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    targets = inputs.clone()

    model = MiniLLM(vocab_size=len(tokenizer.token_to_id))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for _ in range(args.epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

    print("Training complete")


if __name__ == "__main__":
    main()
