"""Evaluation script for MiniLLM."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch


from .model import MiniTransformer, ModelConfig
from .tokenizer import Tokenizer

VOCAB_PATH = Path("data/vocab.json")
MODEL_PATH = Path("data/model.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the MiniLLM model")
    parser.add_argument(
        "--question",
        type=str,
        default="hello world",
        help="question to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(MODEL_PATH),
        help="path to trained model weights",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = Tokenizer()
    if VOCAB_PATH.exists():
        tokenizer.load_vocab(str(VOCAB_PATH))
    else:
        raise FileNotFoundError(f"Vocabulary file not found at {VOCAB_PATH}")

    encoded = tokenizer.encode(args.question, add_bos=True, add_eos=True)
    ids = torch.tensor([encoded], dtype=torch.long)

    config = ModelConfig(
        vocab_size=len(tokenizer.token_to_id),
        emb_dim=32,
        num_heads=4,
        max_seq_len=len(encoded),
        learnable_pos=False,
        ffn_dim=128,
    )
    model = MiniTransformer(config)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(ids)

    pred_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    decoded = tokenizer.decode(pred_ids, skip_special_tokens=True)

    print(decoded)


if __name__ == "__main__":
    main()
