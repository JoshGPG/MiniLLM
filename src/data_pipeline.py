"""Data pipeline utilities for MiniLLM."""

from __future__ import annotations

import json
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

