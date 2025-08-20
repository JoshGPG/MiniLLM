"""Data fetching utilities for MiniLLM."""

from __future__ import annotations

from pathlib import Path
from typing import List

import wikipedia

from .utils import save_text

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
