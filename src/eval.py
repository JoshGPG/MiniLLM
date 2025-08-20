"""Evaluation script for MiniLLM."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch

from .model import MiniLLM, ModelConfig
from .tokenizer import Tokenizer

VOCAB_PATH = Path("data/vocab.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the MiniLLM model")
    parser.add_argument(
        "text",
        nargs="?",
        default="hello world",
        help="input text to evaluate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = Tokenizer()
    if VOCAB_PATH.exists():
        tokenizer.load_vocab(str(VOCAB_PATH))
    else:
        raise FileNotFoundError(f"Vocabulary file not found at {VOCAB_PATH}")

    encoded = tokenizer.encode(args.text, add_bos=True, add_eos=True)
    ids = torch.tensor([encoded], dtype=torch.long)

    config = ModelConfig(
        vocab_size=len(tokenizer.token_to_id),
        emb_dim=32,
        max_seq_len=len(encoded),
        learnable_pos=False,
    )
    model = MiniLLM(config)

    with torch.no_grad():
        logits = model(ids)

    pred_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    decoded = tokenizer.decode(pred_ids, skip_special_tokens=True)

    print("Logits:", logits)
    print("Predicted token ids:", pred_ids)
    print("Decoded prediction:", decoded)


if __name__ == "__main__":
    main()
