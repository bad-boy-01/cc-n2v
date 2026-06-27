"""
lib/agents/story_agent.py — Novel text segmentation agent.

Ports the logic from .claude/agents/novel-to-narration-script.md Step 1.

Reads source novel, splits into ~4-second narration segments at natural
sentence boundaries, marks segment_break at scene transitions.

Output: projects/{name}/drafts/episode_N/step1_segments.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.agents import BaseAgent

# ── Sentence boundary pattern (handles Chinese + English) ─────────────────────
_SENTENCE_ENDS = re.compile(
    r'(?<=[。！？…\.!?])\s*|(?<=\.\.\.)\s*'
)

# Characters per second for Chinese narration (~4 chars/s read aloud)
# English: ~13 chars/s  Chinese: ~4 chars/s
CJK_CHARS_PER_SEC = 4.5
EN_CHARS_PER_SEC = 13.0
DEFAULT_SEGMENT_SECONDS = 4  # target duration per segment
MIN_SEGMENT_CHARS = 10
MAX_SEGMENT_CHARS = 80        # hard cap; long sentences get 6 or 8 seconds


def _is_cjk(text: str) -> bool:
    """Heuristic: if >30% of chars are CJK, treat as Chinese."""
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return cjk_count / max(len(text), 1) > 0.3


def _chars_to_seconds(text: str) -> int:
    """Estimate narration duration, rounded to nearest 2 seconds (4/6/8)."""
    rate = CJK_CHARS_PER_SEC if _is_cjk(text) else EN_CHARS_PER_SEC
    raw = len(text.strip()) / rate
    if raw <= 5:
        return 4
    if raw <= 7:
        return 6
    return 8


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences at natural punctuation boundaries."""
    # Split on sentence-ending punctuation
    parts = re.split(r'(?<=[。！？…\.!?])\s*', text)
    return [p.strip() for p in parts if p.strip()]


def _detect_scene_break(sentences: List[str], idx: int) -> bool:
    """
    Heuristic: mark segment_break if sentence starts with a time/location cue
    or after a dialogue block ending.
    """
    if idx == 0:
        return False
    s = sentences[idx].lower()
    cues = [
        # time shifts
        "次日", "翌日", "翌晨", "数日后", "数月后", "一年后", "meanwhile",
        "the next day", "the following", "years later", "months later",
        "that night", "that evening", "that morning",
        # location shifts
        "另一边", "与此同时", "城外", "皇宫", "elsewhere",
    ]
    return any(s.startswith(c) for c in cues)


def _detect_dialogue(text: str) -> bool:
    """Return True if segment contains quoted dialogue."""
    return bool(re.search(r'[""「」『』\'"]', text))


class StoryAgent(BaseAgent):
    """
    Segments a novel into narration-ready chunks.

    Parameters
    ----------
    project_name : str
    episode : int
        Episode number (1-indexed).
    source_file : str | None
        Filename inside projects/{name}/source/. Auto-detected if None.
    """

    def __init__(
        self,
        project_name: str,
        episode: int = 1,
        source_file: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(project_name, **kwargs)
        self.episode = episode
        self.source_file = source_file

    def _find_source(self) -> Path:
        source_dir = self.get_path("source")
        if self.source_file:
            p = source_dir / self.source_file
            if not p.exists():
                raise FileNotFoundError(f"Source file not found: {p}")
            return p
        # Auto-detect: first .txt or .md file
        candidates = sorted(source_dir.glob("*.txt")) + sorted(source_dir.glob("*.md"))
        if not candidates:
            raise FileNotFoundError(f"No source text found in {source_dir}")
        return candidates[0]

    def _segment_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into ~4-second segments."""
        sentences = _split_sentences(text)
        segments = []
        buffer = ""
        seg_idx = 1

        for i, sent in enumerate(sentences):
            if not sent:
                continue

            # Accumulate until we hit target length
            trial = buffer + sent
            estimated_secs = _chars_to_seconds(trial)

            if len(buffer) > 0 and (
                estimated_secs > DEFAULT_SEGMENT_SECONDS + 2
                or len(trial) > MAX_SEGMENT_CHARS
            ):
                # Flush buffer as a segment
                seg = self._make_segment(buffer, seg_idx, i - 1, sentences)
                segments.append(seg)
                seg_idx += 1
                buffer = sent
            else:
                buffer = trial + ("" if trial.endswith(("。", "！", "？", ".", "!", "?")) else "")

        if buffer.strip():
            seg = self._make_segment(buffer, seg_idx, len(sentences) - 1, sentences)
            segments.append(seg)

        return segments

    def _make_segment(
        self,
        text: str,
        seg_idx: int,
        sentence_idx: int,
        all_sentences: List[str],
    ) -> Dict[str, Any]:
        ep = self.episode
        seg_id = f"E{ep}S{seg_idx:02d}"
        duration = _chars_to_seconds(text)
        return {
            "segment_id": seg_id,
            "episode": ep,
            "sequence": seg_idx,
            "novel_text": text.strip(),
            "char_count": len(text.strip()),
            "duration_seconds": duration,
            "has_dialogue": _detect_dialogue(text),
            "segment_break": _detect_scene_break(all_sentences, sentence_idx),
            "image_prompt": "",          # filled by storyboard_agent
            "camera_motion": "Static",   # filled by storyboard_agent
            "generated_assets": {
                "storyboard_image": None,
                "audio_clip": None,
                "status": "pending",
            },
        }

    def _use_qwen_for_breaks(self, segments: List[Dict]) -> List[Dict]:
        """
        Optional: use Qwen2.5 to improve segment_break detection.
        Only called when the novel is long enough to justify it.
        """
        if len(segments) < 10:
            return segments

        self.log("Using Qwen2.5 to refine scene-break detection …")

        texts = [s["novel_text"] for s in segments]
        prompt = (
            "Below is a numbered list of narration segments from a story. "
            "For each segment number that represents a MAJOR scene transition "
            "(time jump, location change, or chapter break), output its number. "
            "Output ONLY a JSON array of integers. Example: [3, 7, 15]\n\n"
            + "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts[:60]))
        )
        try:
            result = self.qwen.generate_json(prompt, temperature=0.0)
            break_indices = set()
            if isinstance(result, list):
                break_indices = {int(x) - 1 for x in result if isinstance(x, int)}
            elif isinstance(result, dict):
                # Some models return {"breaks": [...]}
                for v in result.values():
                    if isinstance(v, list):
                        break_indices = {int(x) - 1 for x in v}
                        break

            for i in break_indices:
                if 0 <= i < len(segments):
                    segments[i]["segment_break"] = True
        except Exception as e:
            self.log(f"Qwen scene-break detection failed (using heuristics): {e}")

        return segments

    def run(
        self,
        use_qwen_breaks: bool = True,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Run text segmentation.

        Returns
        -------
        dict with keys: segments, output_path, episode
        """
        drafts_dir = self.ensure_dir("drafts", f"episode_{self.episode}")
        output_path = drafts_dir / "step1_segments.json"

        if output_path.exists() and not overwrite:
            self.log(f"Segments already exist at {output_path}. Use overwrite=True to re-run.")
            with open(output_path, encoding="utf-8") as f:
                return {"segments": json.load(f), "output_path": str(output_path), "episode": self.episode}

        source_path = self._find_source()
        self.log(f"Reading source: {source_path.name}")
        text = source_path.read_text(encoding="utf-8")

        segments = self._segment_text(text)
        self.log(f"Initial segmentation: {len(segments)} segments")

        if use_qwen_breaks:
            segments = self._use_qwen_for_breaks(segments)

        break_count = sum(1 for s in segments if s["segment_break"])
        total_secs = sum(s["duration_seconds"] for s in segments)
        self.log(
            f"Final: {len(segments)} segments, {break_count} scene breaks, "
            f"~{total_secs // 60}m {total_secs % 60}s estimated narration"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        self.log(f"Saved to {output_path}")
        return {
            "segments": segments,
            "output_path": str(output_path),
            "episode": self.episode,
            "total_segments": len(segments),
            "estimated_minutes": total_secs // 60,
        }
