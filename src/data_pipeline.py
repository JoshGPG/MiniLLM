"""Utilities for collecting and processing data for MiniLLM.

This module powers the command line interface documented in the README.  It
can fetch short reference articles from Wikipedia, turn them into question and
answer pairs, and create train/validation/test splits saved as JSON Lines.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import wikipedia

from .utils import save_text

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")

# A small but diverse set of topics that mirrors the project scope in the
# README.  Users can override this list via command line flags, but having a
# sensible default makes ``python src/data_pipeline.py fetch`` work out of the
# box.
DEFAULT_TOPICS: Tuple[str, ...] = (
    "Sun",
    "Moon",
    "Mars",
    "Jupiter",
    "Human heart",
    "Human brain",
    "African elephant",
    "Honey bee",
    "Water cycle",
    "Photosynthesis",
    "Hurricane",
    "Plate tectonics",
)


def fetch_articles(topics: Sequence[str]) -> List[Dict[str, str]]:
    """Fetch short Wikipedia articles for the given topics.

    Each article is truncated to the first 300 words and stored under
    ``data/raw/<topic>.txt``.  A JSON compatible representation containing the
    topic and the truncated article text is returned for further processing.
    """

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    articles: List[Dict[str, str]] = []
    for topic in topics:
        try:
            text = wikipedia.page(topic).content
        except Exception:
            # Skip topics that cannot be retrieved, e.g. disambiguation pages or
            # missing entries.
            continue

        words = text.split()
        limited = " ".join(words[:300])
        save_text(RAW_DIR / f"{topic.replace(' ', '_')}.txt", limited)
        articles.append({"topic": topic, "content": limited})

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
        if "context" in pair and pair["context"] is not None:
            item["context"] = clean_text(pair["context"])
        cleaned.append(item)

    if path.suffix == ".json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
    elif path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for item in cleaned:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
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
    output_dir: Path | None = None,
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

    target_dir = output_dir or SPLITS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(train_pairs, target_dir / "train.jsonl")
    save_dataset(val_pairs, target_dir / "val.jsonl")
    save_dataset(test_pairs, target_dir / "test.jsonl")

    return train_pairs, val_pairs, test_pairs


def _load_jsonl(path: Path) -> List[Dict[str, str]]:
    """Load a JSON Lines file into memory."""

    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(records: Iterable[Dict[str, str]], path: Path) -> None:
    """Write an iterable of dictionaries to ``path`` in JSON Lines format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_topics(args: argparse.Namespace) -> List[str]:
    """Resolve topics from command line arguments."""

    topics: List[str] = []
    if getattr(args, "topics_file", None):
        with Path(args.topics_file).open("r", encoding="utf-8") as f:
            topics.extend(line.strip() for line in f if line.strip())
    if getattr(args, "topics", None):
        topics.extend(args.topics)
    if not topics:
        topics = list(DEFAULT_TOPICS)
    return topics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Create the argument parser for the data pipeline CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser(
        "fetch", help="Download reference articles and store them as JSON Lines"
    )
    fetch_parser.add_argument(
        "--output",
        type=Path,
        default=RAW_DIR / "articles.jsonl",
        help="Path where the raw articles JSONL file should be written.",
    )
    fetch_parser.add_argument(
        "--topics",
        nargs="*",
        help="Optional list of topics to fetch. If omitted a default list is used.",
    )
    fetch_parser.add_argument(
        "--topics-file",
        type=Path,
        help="Read topics from a text file (one topic per line).",
    )

    generate_parser = subparsers.add_parser(
        "generate", help="Build question/answer pairs from raw articles"
    )
    generate_parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DIR / "articles.jsonl",
        help="JSONL file produced by the fetch command.",
    )
    generate_parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "qa_pairs.jsonl",
        help="Where to store the generated question/answer pairs.",
    )

    split_parser = subparsers.add_parser(
        "split", help="Create train/validation/test splits from Q&A pairs"
    )
    split_parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DIR / "qa_pairs.jsonl",
        help="JSONL file with question/answer pairs.",
    )
    split_parser.add_argument(
        "--output-dir",
        type=Path,
        default=SPLITS_DIR,
        help="Directory where split files should be stored.",
    )
    split_parser.add_argument("--train-size", type=float, default=0.8)
    split_parser.add_argument("--val-size", type=float, default=0.1)
    split_parser.add_argument("--test-size", type=float, default=0.1)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry-point for the command line interface."""

    args = parse_args(argv)

    if args.command == "fetch":
        topics = _parse_topics(args)
        articles = fetch_articles(topics)
        _write_jsonl(articles, args.output)
        return 0

    if args.command == "generate":
        articles = _load_jsonl(args.input)
        pairs: List[Dict[str, str]] = []
        for article in articles:
            text = article.get("content") or article.get("text") or ""
            if not text:
                continue
            pairs.extend(build_qa_pairs(text))
        save_dataset(pairs, args.output)
        return 0

    if args.command == "split":
        ratios = (args.train_size, args.val_size, args.test_size)
        pairs = _load_jsonl(args.input)
        split_dataset(pairs, ratios, output_dir=args.output_dir)
        return 0

    raise ValueError(f"Unhandled command: {args.command}")


__all__ = [
    "fetch_articles",
    "build_qa_pairs",
    "clean_text",
    "save_dataset",
    "split_dataset",
    "parse_args",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

