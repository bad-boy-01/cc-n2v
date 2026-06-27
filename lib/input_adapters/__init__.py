"""
lib/input_adapters/__init__.py — Input adapter registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from lib.input_adapters.adapters import (
    DOCXAdapter,
    EPUBAdapter,
    ManhwaAdapter,
    MarkdownAdapter,
    NovelAdapter,
    PDFAdapter,
    WebsiteAdapter,
)
from plugins.base import InputAdapter

_ADAPTERS = [
    NovelAdapter(),
    MarkdownAdapter(),
    ManhwaAdapter(),
    PDFAdapter(),
    EPUBAdapter(),
    DOCXAdapter(),
    WebsiteAdapter(),
]


def get_adapter_for_source(source: str) -> Optional[InputAdapter]:
    """
    Find the appropriate adapter for a given input source.
    """
    if source.startswith("http://") or source.startswith("https://"):
        return WebsiteAdapter()

    path = Path(source)
    if path.is_dir():
        # Heuristic: if directory contains images, it's manhwa
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        if any(p.suffix.lower() in exts for p in path.iterdir()):
            return ManhwaAdapter()
        return None

    ext = path.suffix.lower()
    for adapter in _ADAPTERS:
        if ext in adapter.supported_extensions:
            return adapter

    return None
