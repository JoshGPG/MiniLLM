
"""Data pipeline utilities for MiniLLM."""

from __future__ import annotations

import json

import random
from pathlib import Path
from typing import Dict, List, Tuple


def split_dataset(
    pairs: List[Dict[str, str]],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Split a list of pairs into train/validation/test sets.

    Args:
        pairs: List of dictionaries containing paired data.
        ratios: Ratios for train, validation and test splits. Must sum to 1.0.

    Returns:
        A tuple of three lists: ``(train, val, test)``.
    """
    if len(ratios) != 3:
        raise ValueError("ratios must be a tuple of three floats")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")

    rng = random.Random(42)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)
    total = len(shuffled)

    train_end = int(ratios[0] * total)
    val_end = train_end + int(ratios[1] * total)

    train_pairs = shuffled[:train_end]
    val_pairs = shuffled[train_end:val_end]
    test_pairs = shuffled[val_end:]

    # Save splits to disk
    splits_dir = Path("data") / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    with (splits_dir / "train.json").open("w", encoding="utf-8") as f:
        json.dump(train_pairs, f, ensure_ascii=False, indent=2)
    with (splits_dir / "val.json").open("w", encoding="utf-8") as f:
        json.dump(val_pairs, f, ensure_ascii=False, indent=2)
    with (splits_dir / "test.json").open("w", encoding="utf-8") as f:
        json.dump(test_pairs, f, ensure_ascii=False, indent=2)

    return train_pairs, val_pairs, test_pairs


import re
from typing import Iterable, Dict, List


def clean_text(text: str) -> str:
    """Normalize text by lowercasing, collapsing whitespace, and removing non-ASCII chars."""
    # Lowercase the text
    text = text.lower()
    # Remove non-ASCII characters by encoding/decoding
    text = text.encode("ascii", errors="ignore").decode("ascii")
    # Collapse multiple whitespace into single spaces and strip
    text = re.sub(r"\s+", " ", text).strip()
    return text


def save_qa_pairs(pairs: Iterable[Dict[str, str]], path: str) -> None:
    """Save question-answer pairs to ``path`` after cleaning text."""
    cleaned: List[Dict[str, str]] = []
    for pair in pairs:
        question = clean_text(pair.get("question", ""))
        answer = clean_text(pair.get("answer", ""))
        cleaned.append({"question": question, "answer": answer})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

from __future__ import annotations
from pathlib import Path
import re
from typing import Dict, List
import wikipedia

from .utils import save_text

def build_qa_pairs(text: str) -> List[Dict[str, str]]:
    """Convert raw text into question–answer pairs.

    The input text is split into short snippets of 2–4 sentences. Each
    snippet becomes an entry consisting of a simple question, the answer, and
    the original snippet as optional context. The resulting list can later be
    serialized to create training data.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    pairs: List[Dict[str, str]] = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i : i + 3]  # group into ~3 sentences per answer
        if len(chunk) == 1 and pairs:
            pairs[-1]["answer"] += " " + chunk[0]
            pairs[-1]["context"] += " " + chunk[0]
            break
        answer = " ".join(chunk)
        topic = " ".join(chunk[0].split()[:5]).rstrip(",;:?.!")
        question = f"What does the text explain about {topic}?"
        pairs.append({"question": question, "answer": answer, "context": answer})
        i += len(chunk)
    return pairs
"""Data fetching utilities for MiniLLM."""





DATA_DIR = Path("data/raw")


def fetch_articles(topics: List[str]) -> List[str]:
    """Fetch Wikipedia articles for given topics.

    Downloads the article text for each topic, truncates it to a few hundred
    words, saves each to ``data/raw/<topic>.txt``, and returns the collected
    texts.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
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
        filename = DATA_DIR / f"{topic.replace(' ', '_')}.txt"
        save_text(filename, limited)
    return articles
