"""
lib/manhwa_ocr.py — OCR extraction for manhwa/manga chapter images.

Used in --mode manhwa pipeline:
  Chapter Images → OCR → Text Extraction → Story Summarization → ...

Primary:  EasyOCR (GPU-accelerated, MIT license)
Fallback: PaddleOCR

After extraction, text is cleaned by lib/ocr_cleaner.py before Qwen summarization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lib.config import OCR_MIN_CONFIDENCE, DEFAULT_READING_ORDER
from lib.ocr_cleaner import clean_ocr_output, clean_chapter_ocr

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


class ManhwaOCR:
    """
    OCR text extraction from manhwa/manga chapter images.

    Usage
    -----
    ocr = ManhwaOCR(project_name="my_manhwa")
    text = ocr.extract_folder(Path("chapters/ch01/"))
    # text = {"page_001.jpg": "cleaned text ...", ...}
    """

    def __init__(
        self,
        project_name: str,
        projects_root: str = "projects",
        reading_order: str = DEFAULT_READING_ORDER,
        min_confidence: float = OCR_MIN_CONFIDENCE,
        use_gpu: bool = True,
        languages: Optional[List[str]] = None,
    ):
        self.project_name = project_name
        self.project_dir = Path(projects_root) / project_name
        self.reading_order = reading_order
        self.min_confidence = min_confidence
        self.use_gpu = use_gpu
        self.languages = languages or ["en"]  # ["en", "ch_sim"] for Chinese manhwa

        self._reader = None    # EasyOCR reader (lazy)
        self._paddle = None    # PaddleOCR fallback (lazy)
        self._backend: Optional[str] = None

        self.source_dir = self.project_dir / "source"
        self.source_dir.mkdir(parents=True, exist_ok=True)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_easyocr(self) -> bool:
        if self._reader is not None:
            return True
        try:
            import easyocr
            logger.info(f"Loading EasyOCR (gpu={self.use_gpu}, lang={self.languages}) …")
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                verbose=False,
            )
            self._backend = "easyocr"
            logger.info("EasyOCR loaded ✅")
            return True
        except ImportError:
            logger.warning("easyocr not installed — trying PaddleOCR")
            return False
        except Exception as e:
            logger.warning(f"EasyOCR load failed: {e} — trying PaddleOCR")
            return False

    def _load_paddle(self) -> bool:
        if self._paddle is not None:
            return True
        try:
            from paddleocr import PaddleOCR
            lang = "en"
            if any(l in self.languages for l in ["ch_sim", "ch_tra", "chinese"]):
                lang = "ch"
            logger.info(f"Loading PaddleOCR (lang={lang}) …")
            self._paddle = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
            self._backend = "paddleocr"
            logger.info("PaddleOCR loaded ✅")
            return True
        except ImportError:
            logger.error("Neither EasyOCR nor PaddleOCR is installed")
            return False
        except Exception as e:
            logger.error(f"PaddleOCR load failed: {e}")
            return False

    def unload(self) -> None:
        """Release OCR models from memory."""
        import gc
        self._reader = None
        self._paddle = None
        self._backend = None
        gc.collect()

    # ── Page-level OCR ────────────────────────────────────────────────────────

    def _ocr_page_easyocr(self, image_path: Path) -> List[Any]:
        """Run EasyOCR on one page. Returns raw EasyOCR result list."""
        try:
            results = self._reader.readtext(str(image_path), detail=1)
            return results
        except Exception as e:
            logger.warning(f"EasyOCR failed on {image_path.name}: {e}")
            return []

    def _ocr_page_paddle(self, image_path: Path) -> List[Any]:
        """Run PaddleOCR and convert output to EasyOCR-compatible format."""
        try:
            result = self._paddle.ocr(str(image_path), cls=True)
            # PaddleOCR format: [[[bbox], (text, confidence)], ...]
            converted = []
            for line in result[0] or []:
                bbox, (text, conf) = line
                converted.append((bbox, text, conf))
            return converted
        except Exception as e:
            logger.warning(f"PaddleOCR failed on {image_path.name}: {e}")
            return []

    def extract_page(self, image_path: Path) -> str:
        """
        OCR a single page image and return cleaned text.

        Returns empty string if OCR fails.
        """
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return ""

        # Try EasyOCR first
        if self._backend == "easyocr" or self._load_easyocr():
            raw = self._ocr_page_easyocr(image_path)
        elif self._load_paddle():
            raw = self._ocr_page_paddle(image_path)
        else:
            logger.error("No OCR backend available")
            return ""

        # Fallback to Paddle if EasyOCR gives empty result
        if not raw and self._backend == "easyocr" and self._load_paddle():
            logger.debug(f"  EasyOCR empty for {image_path.name}, trying PaddleOCR")
            raw = self._ocr_page_paddle(image_path)

        return clean_ocr_output(
            raw,
            min_confidence=self.min_confidence,
            reading_order=self.reading_order,
        )

    # ── Chapter-level extraction ──────────────────────────────────────────────

    def extract_chapter(self, image_paths: List[Path]) -> str:
        """
        OCR all pages in a chapter and return combined cleaned text.

        Parameters
        ----------
        image_paths : list of Paths, sorted in reading order

        Returns
        -------
        str : full chapter text ready for Qwen2.5 summarization
        """
        pages: Dict[str, List[Any]] = {}

        for img_path in image_paths:
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            logger.debug(f"  OCR: {img_path.name}")

            if self._backend == "easyocr" or self._load_easyocr():
                raw = self._ocr_page_easyocr(img_path)
            elif self._load_paddle():
                raw = self._ocr_page_paddle(img_path)
            else:
                raw = []

            pages[img_path.name] = raw

        return clean_chapter_ocr(pages, reading_order=self.reading_order)

    def extract_folder(self, folder: Path) -> Dict[str, str]:
        """
        OCR all image pages in a folder, returning per-page cleaned text.

        Also saves combined text to:
            projects/{name}/source/{folder.name}_ocr.txt

        Returns
        -------
        dict: {page_filename: cleaned_text}
        """
        if not folder.exists():
            raise FileNotFoundError(f"Chapter folder not found: {folder}")

        image_files = sorted([
            f for f in folder.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ])

        if not image_files:
            raise ValueError(f"No images found in {folder}")

        logger.info(f"[ManhwaOCR] Processing {len(image_files)} pages from {folder.name} …")

        page_texts: Dict[str, str] = {}
        for img_path in image_files:
            text = self.extract_page(img_path)
            if text:
                page_texts[img_path.name] = text
                logger.info(f"  {img_path.name}: {len(text)} chars")
            else:
                logger.warning(f"  {img_path.name}: no text extracted")

        # Save combined chapter text
        combined = "\n\n".join(page_texts.values())
        out_path = self.source_dir / f"{folder.name}_ocr.txt"
        out_path.write_text(combined, encoding="utf-8")
        logger.info(f"[ManhwaOCR] Saved to {out_path} ({len(combined)} chars total)")

        return page_texts

    # ── Multi-chapter extraction ──────────────────────────────────────────────

    def extract_all_chapters(self, chapters_root: Path) -> Dict[str, str]:
        """
        Extract text from all chapter subfolders.

        Expected structure:
            chapters_root/
                chapter_001/
                    page_001.jpg
                    page_002.jpg
                chapter_002/
                    ...

        Returns
        -------
        dict: {chapter_name: full_chapter_text}
        """
        if not chapters_root.exists():
            raise FileNotFoundError(f"Chapters root not found: {chapters_root}")

        chapter_dirs = sorted([
            d for d in chapters_root.iterdir()
            if d.is_dir()
        ])

        if not chapter_dirs:
            # Flat folder — treat entire folder as one chapter
            logger.info("[ManhwaOCR] Flat folder detected — treating as single chapter")
            texts = self.extract_folder(chapters_root)
            return {"chapter_001": "\n\n".join(texts.values())}

        logger.info(f"[ManhwaOCR] Found {len(chapter_dirs)} chapters")
        chapter_texts: Dict[str, str] = {}

        for ch_dir in chapter_dirs:
            out_path = self.source_dir / f"{ch_dir.name}_ocr.txt"
            if out_path.exists():
                logger.info(f"  {ch_dir.name}: OCR file exists, loading cached")
                chapter_texts[ch_dir.name] = out_path.read_text(encoding="utf-8")
                continue

            try:
                page_texts = self.extract_folder(ch_dir)
                chapter_texts[ch_dir.name] = "\n\n".join(page_texts.values())
            except Exception as e:
                logger.error(f"  {ch_dir.name}: failed — {e}")

        # Save master index
        index_path = self.source_dir / "chapters_index.json"
        summary = {name: len(text) for name, text in chapter_texts.items()}
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return chapter_texts
