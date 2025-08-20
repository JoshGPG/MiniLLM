
"""Data pipeline utilities for saving datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def save_dataset(pairs: List[Dict[str, str]], path: Path) -> None:
    """Save question/answer pairs to JSON or CSV.

    The file is written inside ``data/processed`` using the file name from
    ``path``. Supported formats are JSON (``.json``) and CSV (``.csv``).

    Args:
        pairs: Sequence of dictionaries with ``question`` and ``answer`` keys and
            an optional ``context`` key.
        path: Target file path. Only the file name is used; the directory is
            always ``data/processed``.
    """
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    final_path = processed_dir / path.name

    items = []
    has_context = False
    for pair in pairs:
        item = {"question": pair["question"], "answer": pair["answer"]}
        if "context" in pair and pair["context"] is not None:
            item["context"] = pair["context"]
            has_context = True
        items.append(item)

    if final_path.suffix == ".json":
        with final_path.open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    elif final_path.suffix == ".csv":
        fieldnames = ["question", "answer"] + (["context"] if has_context else [])
        with final_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in items:
                writer.writerow(row)
    else:
        raise ValueError(f"Unsupported file format: {final_path.suffix}")
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
