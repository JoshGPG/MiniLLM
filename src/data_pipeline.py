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
