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

