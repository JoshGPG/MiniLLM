"""Text generation script for MiniLLM.

This module loads a trained language model and produces text from a given
prompt. Generation uses greedy decoding when ``temperature`` is set to ``0``
otherwise sampling with temperature scaling is applied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import MiniTransformer, ModelConfig
from .tokenizer import Tokenizer


VOCAB_PATH = Path("data/vocab.json")
MODEL_PATH = Path("data/model.pt")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate text with MiniLLM")
    parser.add_argument(
        "prompt", nargs="?", default="hello", help="seed text to start generation"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=20,
        help="number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="sampling temperature; 0 for greedy decoding",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(MODEL_PATH),
        help="path to trained model weights",
    )
    return parser.parse_args()


def sample(
    model: MiniTransformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generate text from ``prompt`` using ``model`` and ``tokenizer``."""

    model.eval()
    device = next(model.parameters()).device

    generated = tokenizer.encode(prompt, add_bos=True)
    input_ids = torch.tensor([generated], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)[:, -1, :]
        if temperature <= 0:
            next_id = int(torch.argmax(logits, dim=-1))
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1))
        generated.append(next_id)
        if next_id == tokenizer.eos_id:
            break
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)

    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    args = parse_args()

    tokenizer = Tokenizer()
    if VOCAB_PATH.exists():
        tokenizer.load_vocab(str(VOCAB_PATH))
    else:
        raise FileNotFoundError(f"Vocabulary file not found at {VOCAB_PATH}")

    config = ModelConfig(vocab_size=len(tokenizer.token_to_id))
    model = MiniTransformer(config)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    text = sample(model, tokenizer, args.prompt, args.seq_len, args.temperature)
    print(text)


if __name__ == "__main__":
    main()
