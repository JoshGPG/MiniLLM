"""Utility functions for MiniLLM."""

from __future__ import annotations
from pathlib import Path


def read_text(path: Path) -> str:
    """Read text from a file."""
    return path.read_text(encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    """Save text to a file."""
    path.write_text(text, encoding="utf-8")
