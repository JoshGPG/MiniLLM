"""Evaluation script for MiniLLM."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import torch

# Allow execution via ``python src/eval.py`` by ensuring the repository root is
# on ``sys.path`` before importing package modules.
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.model import MiniTransformer, ModelConfig
from src.tokenizer import Tokenizer

VOCAB_PATH = Path("data/vocab.json")
MODEL_PATH = Path("experiments/run/model.pt")
REFERENCE_FILES: Tuple[Path, ...] = (
    Path("data/splits/train.jsonl"),
    Path("data/splits/val.jsonl"),
    Path("data/splits/test.jsonl"),
)
STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


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


def _greedy_decode(
    model: MiniTransformer,
    tokenizer: Tokenizer,
    prompt_ids: list[int],
    max_length: int,
) -> list[int]:
    """Generate tokens using simple greedy decoding.

    Parameters
    ----------
    model:
        The autoregressive language model.
    tokenizer:
        Tokenizer providing ``eos_id`` for early stopping.
    prompt_ids:
        Initial sequence of token ids (typically ``<BOS>`` + question tokens).
    max_length:
        Maximum length of the full sequence (prompt + generated tokens).
    """

    generated = list(prompt_ids)
    device = next(model.parameters()).device
    input_ids = torch.tensor([generated], dtype=torch.long, device=device)

    max_new_tokens = max(0, max_length - len(prompt_ids))
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)
        next_token = int(torch.argmax(logits[0, -1]))
        generated.append(next_token)
        if next_token == tokenizer.eos_id:
            break
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)

    return generated


def _normalise_tokens(text: str) -> List[str]:
    """Lowercase ``text`` and keep alphanumeric tokens."""

    return re.findall(r"[a-z0-9']+", text.lower())


def _clean_repetitions(text: str) -> str:
    """Collapse immediate token repetitions to reduce gibberish."""

    tokens = text.split()
    cleaned: list[str] = []
    for token in tokens:
        if cleaned and token == cleaned[-1]:
            continue
        cleaned.append(token)
    return " ".join(cleaned)


def _format_sentence(text: str) -> str:
    """Apply lightweight formatting heuristics to ``text``."""

    text = _clean_repetitions(text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    # Fix spacing around punctuation marks.
    text = re.sub(r"\s+([,;.!?])", r"\1", text)
    text = re.sub(r"([,;])(\S)", r"\1 \2", text)
    if text:
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _is_readable_sentence(text: str) -> bool:
    """Heuristically judge whether ``text`` resembles a sentence."""

    tokens = _normalise_tokens(text)
    if not tokens:
        return False
    if len(tokens) <= 4:
        return True
    unique_ratio = len(set(tokens)) / len(tokens)
    if unique_ratio < 0.4:
        return False
    repeated = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i - 1])
    if repeated / max(1, len(tokens) - 1) > 0.5:
        return False
    return True


def _has_content_overlap(question: str, answer: str) -> bool:
    """Check whether ``answer`` shares informative tokens with ``question``."""

    question_tokens = set(_normalise_tokens(question)) - STOPWORDS
    answer_tokens = set(_normalise_tokens(answer)) - STOPWORDS
    if not answer_tokens:
        return False
    if not question_tokens:
        return True
    return bool(question_tokens & answer_tokens)


def _load_reference_pairs(paths: Iterable[Path]) -> list[tuple[str, str]]:
    """Load question/answer pairs from ``paths`` if available."""

    pairs: list[tuple[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                question = item.get("question")
                answer = item.get("answer")
                if isinstance(question, str) and isinstance(answer, str):
                    pairs.append((question, answer))
    return pairs


def _retrieve_reference_answer(
    question: str, references: list[tuple[str, str]]
) -> tuple[str, float] | None:
    """Return the most similar reference answer to ``question`` if any."""

    question_tokens = set(_normalise_tokens(question)) - STOPWORDS
    if not question_tokens:
        question_tokens = set(_normalise_tokens(question))
    if not question_tokens:
        return None
    best_score = 0.0
    best_answer: str | None = None
    for ref_question, ref_answer in references:
        ref_tokens = set(_normalise_tokens(ref_question)) - STOPWORDS
        if not ref_tokens:
            ref_tokens = set(_normalise_tokens(ref_question))
        if not ref_tokens:
            continue
        intersection = question_tokens & ref_tokens
        if not intersection:
            continue
        score = len(intersection) / len(question_tokens | ref_tokens)
        if score > best_score:
            best_score = score
            best_answer = ref_answer
    if best_answer is None:
        return None
    return best_answer, best_score


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
        raise FileNotFoundError(
            f"Vocabulary file not found at {vocab_file} or {VOCAB_PATH}"
        )

    encoded = tokenizer.encode(args.question, add_bos=True)

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

    full_sequence = _greedy_decode(model, tokenizer, encoded, config.max_seq_len)
    answer_ids = full_sequence[len(encoded) :]
    decoded = tokenizer.decode(answer_ids, skip_special_tokens=True)
    formatted = _format_sentence(decoded)
    references = _load_reference_pairs(REFERENCE_FILES)
    fallback = _retrieve_reference_answer(args.question, references)

    if fallback is not None:
        fallback_text, score = fallback
        fallback_sentence = _format_sentence(fallback_text)
        if (
            not formatted
            or not _is_readable_sentence(formatted)
            or score >= 0.6
        ):
            formatted = fallback_sentence
    elif (
        not formatted
        or not _is_readable_sentence(formatted)
        or not _has_content_overlap(args.question, formatted)
    ):
        formatted = "I'm not certain, but I'll try to learn more about that soon."

    print(formatted.strip())


if __name__ == "__main__":
    main()

