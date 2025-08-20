"""Utilities for collecting and processing data for MiniLLM."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import wikipedia

from .utils import save_text

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")


def fetch_articles(topics: List[str]) -> List[str]:
    """Fetch short Wikipedia articles for the given topics.

    Each article is truncated to the first 300 words and stored under
    ``data/raw/<topic>.txt``. The collected article texts are returned as a
    list.
    """

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    articles: List[str] = []
    for topic in topics:
        try:
            text = wikipedia.page(topic).content
        except Exception:
            # Skip topics that cannot be retrieved
            continue
        words = text.split()
        limited = " ".join(words[:300])
        articles.append(limited)
        save_text(RAW_DIR / f"{topic.replace(' ', '_')}.txt", limited)
    return articles


def build_qa_pairs(text: str) -> List[Dict[str, str]]:
    """Convert raw article text into question–answer pairs."""

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    pairs: List[Dict[str, str]] = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i : i + 3]
        answer = " ".join(chunk)
        topic = " ".join(chunk[0].split()[:5]).rstrip(",;:?.!")
        question = f"What does the text explain about {topic}?"
        pairs.append({"question": question, "answer": answer, "context": answer})
        i += len(chunk)
    return pairs


def clean_text(text: str) -> str:
    """Normalize text by lowercasing and removing noise."""

    text = text.lower()
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def save_dataset(pairs: Iterable[Dict[str, str]], path: Path) -> None:
    """Save question–answer pairs to ``path`` after cleaning text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned: List[Dict[str, str]] = []
    for pair in pairs:
        item = {
            "question": clean_text(pair.get("question", "")),
            "answer": clean_text(pair.get("answer", "")),
        }
        if "context" in pair:
            item["context"] = clean_text(pair["context"])
        cleaned.append(item)

    if path.suffix == ".json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
    elif path.suffix == ".csv":
        import csv

        fieldnames = ["question", "answer"] + (
            ["context"] if any("context" in p for p in cleaned) else []
        )
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cleaned)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def split_dataset(
    pairs: List[Dict[str, str]],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Split ``pairs`` into train, validation, and test sets and save them."""

    if len(ratios) != 3:
        raise ValueError("ratios must be a tuple of three floats")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")

    rng = random.Random(42)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    total = len(shuffled)

    train_end = int(ratios[0] * total)
    val_end = train_end + int(ratios[1] * total)

    train_pairs = shuffled[:train_end]
    val_pairs = shuffled[train_end:val_end]
    test_pairs = shuffled[val_end:]

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    save_dataset(train_pairs, SPLITS_DIR / "train.json")
    save_dataset(val_pairs, SPLITS_DIR / "val.json")
    save_dataset(test_pairs, SPLITS_DIR / "test.json")

    return train_pairs, val_pairs, test_pairs


__all__ = [
    "fetch_articles",
    "build_qa_pairs",
    "clean_text",
    "save_dataset",
    "split_dataset",
]
