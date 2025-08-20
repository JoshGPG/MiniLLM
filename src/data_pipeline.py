from __future__ import annotations

import re
from typing import Dict, List


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
