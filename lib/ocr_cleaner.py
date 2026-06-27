"""
lib/ocr_cleaner.py — OCR output cleanup for manhwa/manga mode.

Runs between EasyOCR output and Qwen2.5 summarization.
This module has MORE impact on recap quality than switching LLMs.

Pipeline position:
  EasyOCR
    ↓
  ocr_cleaner.clean_ocr_output()   ← THIS FILE
    ↓
  Qwen2.5 summarization
    ↓
  Storyboard
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


# ── Data types ────────────────────────────────────────────────────────────────

class OcrLine:
    """One OCR detection result from EasyOCR."""
    def __init__(self, text: str, confidence: float, bbox: Optional[List] = None):
        self.text = text.strip()
        self.confidence = confidence
        self.bbox = bbox or []
        self.y_center = self._compute_y_center()

    def _compute_y_center(self) -> float:
        if self.bbox and len(self.bbox) >= 2:
            try:
                ys = [pt[1] for pt in self.bbox]
                return sum(ys) / len(ys)
            except (TypeError, IndexError):
                pass
        return 0.0

    def __repr__(self):
        return f"OcrLine({self.text!r}, conf={self.confidence:.2f})"


# ── Individual cleaning functions ─────────────────────────────────────────────

def filter_low_confidence(
    lines: List[OcrLine],
    min_confidence: float = 0.4,
) -> List[OcrLine]:
    """
    Remove lines below confidence threshold.
    EasyOCR confidence range: 0.0 → 1.0
    """
    return [l for l in lines if l.confidence >= min_confidence]


def remove_duplicates(lines: List[OcrLine]) -> List[OcrLine]:
    """
    Remove exact duplicate text lines (same text appearing multiple times
    from overlapping detection regions).
    """
    seen: set = set()
    result = []
    for line in lines:
        key = _normalize_for_dedup(line.text)
        if key and key not in seen:
            seen.add(key)
            result.append(line)
    return result


def remove_repeated_bubbles(
    lines: List[OcrLine],
    similarity_threshold: float = 0.85,
) -> List[OcrLine]:
    """
    Remove near-duplicate speech bubbles.
    Handles cases like:
      "你好" (bubble 1)  
      "你好！" (bubble 2 — same with punctuation)

    Uses character-level Jaccard similarity.
    """
    result: List[OcrLine] = []
    for line in lines:
        normalized = _normalize_for_dedup(line.text)
        if not normalized:
            continue
        is_dup = False
        for existing in result:
            existing_norm = _normalize_for_dedup(existing.text)
            if _jaccard_similarity(normalized, existing_norm) >= similarity_threshold:
                # Keep the higher-confidence version
                if line.confidence > existing.confidence:
                    result.remove(existing)
                    result.append(line)
                is_dup = True
                break
        if not is_dup:
            result.append(line)
    return result


def merge_broken_lines(
    lines: List[OcrLine],
    max_gap_pixels: float = 25.0,
) -> List[OcrLine]:
    """
    Merge lines that are vertically close and likely belong to the same
    speech bubble (broken across multiple OCR detections).

    max_gap_pixels: maximum vertical gap between lines to merge.
    """
    if not lines:
        return lines

    # Sort by vertical position
    sorted_lines = sorted(lines, key=lambda l: l.y_center)
    merged: List[OcrLine] = []
    current = sorted_lines[0]

    for next_line in sorted_lines[1:]:
        gap = next_line.y_center - current.y_center
        if gap <= max_gap_pixels and _can_merge(current.text, next_line.text):
            # Merge: join texts with space or Chinese connector
            joiner = "" if _is_cjk(current.text) else " "
            merged_text = current.text.rstrip("…") + joiner + next_line.text.lstrip("…")
            avg_conf = (current.confidence + next_line.confidence) / 2
            current = OcrLine(merged_text, avg_conf, current.bbox)
        else:
            merged.append(current)
            current = next_line

    merged.append(current)
    return merged


def sort_panel_order(
    lines: List[OcrLine],
    reading_order: str = "manga",
) -> List[OcrLine]:
    """
    Sort OCR lines by reading order.

    reading_order:
      "manga"   — right-to-left, top-to-bottom (Japanese manga)
      "manhwa"  — left-to-right, top-to-bottom (Korean manhwa / webtoon)
    """
    if not lines:
        return lines

    # Group into rows by vertical proximity (30px tolerance)
    rows: List[List[OcrLine]] = []
    sorted_by_y = sorted(lines, key=lambda l: l.y_center)

    current_row = [sorted_by_y[0]]
    for line in sorted_by_y[1:]:
        if abs(line.y_center - current_row[-1].y_center) <= 30:
            current_row.append(line)
        else:
            rows.append(current_row)
            current_row = [line]
    rows.append(current_row)

    # Sort within each row
    result = []
    for row in rows:
        if reading_order == "manga":
            # Right to left
            row_sorted = sorted(row, key=lambda l: _x_center(l), reverse=True)
        else:
            # Left to right
            row_sorted = sorted(row, key=lambda l: _x_center(l))
        result.extend(row_sorted)

    return result


def remove_sfx_and_noise(lines: List[OcrLine]) -> List[OcrLine]:
    """
    Remove sound effects, single characters, and noise tokens
    that don't contribute to narrative content.
    """
    result = []
    for line in lines:
        text = line.text.strip()
        # Skip very short (1-2 chars) unless they're CJK dialogue
        if len(text) <= 2 and not _is_meaningful_short(text):
            continue
        # Skip all-uppercase short words (likely SFX in English)
        if len(text) <= 6 and text.isupper() and text.isalpha():
            continue
        # Skip common SFX patterns
        if _is_sfx(text):
            continue
        result.append(line)
    return result


# ── Main entry point ──────────────────────────────────────────────────────────

def clean_ocr_output(
    raw_results: List[Any],
    min_confidence: float = 0.4,
    reading_order: str = "manhwa",
    merge_lines: bool = True,
    remove_noise: bool = True,
) -> str:
    """
    Full OCR cleanup pipeline.

    Parameters
    ----------
    raw_results : list
        EasyOCR output: list of (bbox, text, confidence) tuples.
    min_confidence : float
        Minimum confidence to keep a detection.
    reading_order : str
        "manhwa" (left-to-right) or "manga" (right-to-left).
    merge_lines : bool
        Whether to merge broken bubble lines.
    remove_noise : bool
        Whether to remove SFX and noise tokens.

    Returns
    -------
    str : cleaned text ready for Qwen2.5 summarization
    """
    # Parse EasyOCR format
    lines = []
    for item in raw_results:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            bbox = item[0] if len(item) >= 3 else []
            text = item[1] if len(item) >= 3 else item[0]
            confidence = float(item[2]) if len(item) >= 3 else float(item[1]) if isinstance(item[1], float) else 1.0
            if isinstance(text, str) and text.strip():
                lines.append(OcrLine(text, confidence, bbox))

    if not lines:
        return ""

    # Apply cleanup pipeline
    lines = filter_low_confidence(lines, min_confidence)
    lines = remove_duplicates(lines)
    lines = remove_repeated_bubbles(lines)

    if merge_lines:
        lines = merge_broken_lines(lines)

    lines = sort_panel_order(lines, reading_order)

    if remove_noise:
        lines = remove_sfx_and_noise(lines)

    # Final dedup after merge
    lines = remove_duplicates(lines)

    return "\n".join(l.text for l in lines if l.text)


def clean_chapter_ocr(
    pages: Dict[str, List[Any]],
    reading_order: str = "manhwa",
) -> str:
    """
    Clean OCR output for an entire chapter (multiple pages).

    Parameters
    ----------
    pages : dict
        {page_filename: easyocr_results}

    Returns
    -------
    str : complete cleaned text for the chapter
    """
    chapter_text = []
    for page_name in sorted(pages.keys()):
        page_text = clean_ocr_output(
            pages[page_name],
            reading_order=reading_order,
        )
        if page_text:
            chapter_text.append(page_text)

    return "\n\n".join(chapter_text)


# ── Private helpers ───────────────────────────────────────────────────────────

def _normalize_for_dedup(text: str) -> str:
    """Normalize text for deduplication comparison."""
    # Lowercase, strip punctuation and whitespace
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\s\.,!?。！？…、，]+", "", text.lower())
    return text


def _jaccard_similarity(a: str, b: str) -> float:
    """Character-level Jaccard similarity."""
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def _is_cjk(text: str) -> bool:
    return sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\uac00' <= c <= '\ud7a3') > len(text) * 0.3


def _is_meaningful_short(text: str) -> bool:
    """True if a 1-2 char string is meaningful (CJK exclamation etc.)."""
    meaningful = {"哦", "啊", "嗯", "哈", "呀", "哎", "喔", "咦"}
    return text in meaningful


def _is_sfx(text: str) -> bool:
    """Detect sound effect text patterns."""
    sfx_patterns = [
        r"^[!！?？]+$",
        r"^[哈嘿嘻呵]+$",
        r"^(BAM|POW|BANG|CRACK|BOOM|SMASH|SLASH|ZAP|WHOOSH|THUD|WHAM)$",
        r"^[\*\-_=~]+$",
    ]
    for pattern in sfx_patterns:
        if re.match(pattern, text.strip(), re.IGNORECASE):
            return True
    return False


def _can_merge(text_a: str, text_b: str) -> bool:
    """True if two adjacent lines should be merged."""
    # Don't merge if first ends with sentence-ending punctuation
    if text_a.rstrip().endswith(("。", "！", "？", ".", "!", "?")):
        return False
    return True


def _x_center(line: OcrLine) -> float:
    """Compute horizontal center of an OCR line's bounding box."""
    if line.bbox and len(line.bbox) >= 2:
        try:
            xs = [pt[0] for pt in line.bbox]
            return sum(xs) / len(xs)
        except (TypeError, IndexError):
            pass
    return 0.0
