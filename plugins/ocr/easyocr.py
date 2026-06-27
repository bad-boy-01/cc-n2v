"""plugins/ocr/easyocr.py — EasyOCR plugin wrapper."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from plugins.base import OCRPlugin


class Plugin(OCRPlugin):
    def __init__(self):
        self._reader = None
        self._languages = None

    def extract(self, image_path, languages=None) -> str:
        langs = languages or ["en"]
        if self._reader is None or langs != self._languages:
            import easyocr
            self._reader = easyocr.Reader(langs, gpu=True, verbose=False)
            self._languages = langs

        results = self._reader.readtext(str(image_path))
        return " ".join(text for _, text, conf in results if conf > 0.5)

    def unload(self) -> None:
        self._reader = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @property
    def plugin_id(self) -> str:
        return "easyocr"
