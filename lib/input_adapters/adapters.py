"""
lib/input_adapters/adapters.py — Input adapters for various source formats.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

from plugins.base import InputAdapter

logger = logging.getLogger(__name__)


class NovelAdapter(InputAdapter):
    """Adapter for plain text novel files (.txt)."""

    def read(self, source: str) -> str:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        return path.read_text(encoding="utf-8", errors="ignore")

    @property
    def supported_extensions(self) -> List[str]:
        return [".txt"]

    @property
    def adapter_id(self) -> str:
        return "novel"


class MarkdownAdapter(InputAdapter):
    """Adapter for Markdown files (.md)."""

    def read(self, source: str) -> str:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Very basic markdown stripping (remove headers, bold, italics, links)
        text = re.sub(r'#+\s+', '', text)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        return text

    @property
    def supported_extensions(self) -> List[str]:
        return [".md", ".markdown"]

    @property
    def adapter_id(self) -> str:
        return "markdown"


class ManhwaAdapter(InputAdapter):
    """Adapter for Manhwa/Manga image directories or CBZ files (extracts via OCR)."""

    def __init__(self):
        self._ocr = None

    def read(self, source: str) -> str:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        image_files = []
        if path.is_dir():
            image_files = sorted(
                p for p in path.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
        else:
            # Assume CBZ/ZIP — would extract to temp dir, keeping it simple for now
            raise NotImplementedError("CBZ support not yet implemented. Use directory of images.")

        if not image_files:
            return ""

        from plugins import load_ocr_plugin
        if not self._ocr:
            self._ocr = load_ocr_plugin("easyocr")

        texts = []
        for img in image_files:
            logger.info(f"Extracting text from {img.name}...")
            text = self._ocr.extract(img)
            if text:
                texts.append(text)
        
        return "\n\n".join(texts)

    @property
    def supported_extensions(self) -> List[str]:
        return [".jpg", ".jpeg", ".png", ".webp", ".cbz", ".zip", ""]

    @property
    def adapter_id(self) -> str:
        return "manhwa"


class PDFAdapter(InputAdapter):
    """Adapter for PDF files using PyMuPDF (fitz)."""

    def read(self, source: str) -> str:
        try:
            import fitz
        except ImportError:
            raise ImportError("Install PyMuPDF: pip install PyMuPDF")

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        doc = fitz.open(str(path))
        texts = []
        for page in doc:
            texts.append(page.get_text())
        return "\n".join(texts)

    @property
    def supported_extensions(self) -> List[str]:
        return [".pdf"]

    @property
    def adapter_id(self) -> str:
        return "pdf"


class EPUBAdapter(InputAdapter):
    """Adapter for EPUB files using ebooklib and BeautifulSoup."""

    def read(self, source: str) -> str:
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("Install dependencies: pip install EbookLib beautifulsoup4")

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        book = epub.read_epub(str(path))
        texts = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_body_content(), "html.parser")
                texts.append(soup.get_text(separator="\n"))
        return "\n".join(texts)

    @property
    def supported_extensions(self) -> List[str]:
        return [".epub"]

    @property
    def adapter_id(self) -> str:
        return "epub"


class DOCXAdapter(InputAdapter):
    """Adapter for DOCX files using python-docx."""

    def read(self, source: str) -> str:
        try:
            import docx
        except ImportError:
            raise ImportError("Install python-docx: pip install python-docx")

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        doc = docx.Document(str(path))
        return "\n".join(para.text for para in doc.paragraphs)

    @property
    def supported_extensions(self) -> List[str]:
        return [".docx"]

    @property
    def adapter_id(self) -> str:
        return "docx"


class WebsiteAdapter(InputAdapter):
    """Adapter for reading web articles (e.g. RoyalRoad, WebNovel) using Newspaper3k."""

    def read(self, source: str) -> str:
        if not (source.startswith("http://") or source.startswith("https://")):
            raise ValueError(f"Invalid URL: {source}")

        try:
            from newspaper import Article
        except ImportError:
            raise ImportError("Install newspaper3k: pip install newspaper3k")

        article = Article(source)
        article.download()
        article.parse()
        return article.text

    @property
    def supported_extensions(self) -> List[str]:
        return []

    @property
    def adapter_id(self) -> str:
        return "website"
