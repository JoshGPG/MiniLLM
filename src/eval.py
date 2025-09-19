"""Evaluation script for MiniLLM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json
import torch

# Allow execution via ``python src/eval.py`` by ensuring the repository root is
# on ``sys.path`` before importing package modules.
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.model import MiniTransformer, ModelConfig
from src.tokenizer import Tokenizer

VOCAB_PATH = Path("data/vocab.json")
MODEL_PATH = Path("experiments/run/model.pt")


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

    checkpoint_path = Path(args.checkpoint)
    run_dir = checkpoint_path.parent

    tokenizer = Tokenizer()
    vocab_file = run_dir / "vocab.json"
    if vocab_file.exists():
        tokenizer.load_vocab(str(vocab_file))
    elif VOCAB_PATH.exists():
        tokenizer.load_vocab(str(VOCAB_PATH))
    else:
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_file} or {VOCAB_PATH}")

    encoded = tokenizer.encode(args.question, add_bos=True, add_eos=True)
    ids = torch.tensor([encoded], dtype=torch.long)

    config_path = run_dir / "model_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        config = ModelConfig(**cfg_dict)
        if len(encoded) > config.max_seq_len:
            config.max_seq_len = len(encoded)
    else:
        config = ModelConfig(
            vocab_size=len(tokenizer.token_to_id),
            emb_dim=32,
            num_layers=2,
            num_heads=2,
            max_seq_len=len(encoded),
            ffn_dim=64,
        )

    model = MiniTransformer(config)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(ids)

    pred_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    decoded = tokenizer.decode(pred_ids, skip_special_tokens=True)

    print(decoded)


if __name__ == "__main__":
    main()

