"""
lib/agents/narration_agent.py — Prepares narration text for TTS.

Reads novel_text from each storyboard segment and optionally uses
Qwen2.5 to rewrite for cleaner TTS output (removing prose artifacts,
handling dialogue attribution, etc.).

This agent does NOT call TTS — that happens in Stage 3 (kokoro_tts.py).
This agent only produces clean text files ready for synthesis.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.agents import BaseAgent

NARRATION_CLEANUP_PROMPT = """
Rewrite the following text for audio narration. Rules:
1. Keep all original content — do not add or remove plot events
2. Remove parenthetical stage directions if any
3. Clarify ambiguous pronouns if the referent is clear from context
4. Split overly long sentences at natural breathing points
5. Preserve all dialogue exactly as written
6. Output ONLY the cleaned narration text, no commentary

Text:
{text}
""".strip()


def _basic_clean(text: str) -> str:
    """Lightweight cleanup that doesn't require LLM."""
    # Remove markdown formatting
    text = re.sub(r"[*_`#]+", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove chapter headings like "Chapter 1:" at start
    text = re.sub(r"^(chapter|第|卷|章)\s*[\d一二三四五六七八九十百]+[：:。\s]*", "", text, flags=re.IGNORECASE)
    return text


class NarrationAgent(BaseAgent):
    """
    Prepares narration text for TTS.

    For most segments, basic_clean() is sufficient.
    use_qwen_rewrite=True enables LLM-assisted rewriting for problem segments.
    """

    def __init__(
        self,
        project_name: str,
        episode: int = 1,
        use_qwen_rewrite: bool = False,
        **kwargs,
    ):
        super().__init__(project_name, **kwargs)
        self.episode = episode
        self.use_qwen_rewrite = use_qwen_rewrite

    def _load_storyboard(self) -> Dict:
        path = self.get_path("scripts", f"episode_{self.episode}.json")
        if not path.exists():
            raise FileNotFoundError(f"Run StoryboardAgent first. Missing: {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _clean_segment(self, seg: Dict) -> str:
        text = seg.get("novel_text", "")
        text = _basic_clean(text)

        if self.use_qwen_rewrite and len(text) > 100:
            try:
                prompt = NARRATION_CLEANUP_PROMPT.format(text=text)
                rewritten = self.qwen.generate(prompt, temperature=0.3)
                if rewritten and len(rewritten) > 20:
                    text = rewritten.strip()
            except Exception as e:
                self.log(f"Qwen rewrite failed for {seg.get('segment_id')}: {e}")

        return text

    def run(self, overwrite: bool = False) -> Dict[str, Any]:
        """
        Write per-segment narration text files to:
        projects/{name}/drafts/episode_N/narration/

        Returns
        -------
        dict: narration_dir, segment_count, text_map {segment_id: cleaned_text}
        """
        script = self._load_storyboard()
        segments = script.get("segments", [])

        narr_dir = self.ensure_dir("drafts", f"episode_{self.episode}", "narration")
        text_map: Dict[str, str] = {}

        for seg in segments:
            sid = seg["segment_id"]
            out_path = narr_dir / f"{sid}.txt"

            if out_path.exists() and not overwrite:
                text_map[sid] = out_path.read_text(encoding="utf-8")
                continue

            cleaned = self._clean_segment(seg)
            out_path.write_text(cleaned, encoding="utf-8")
            text_map[sid] = cleaned

        self.log(f"Narration text ready for {len(text_map)} segments in {narr_dir}")
        return {
            "narration_dir": str(narr_dir),
            "segment_count": len(text_map),
            "text_map": text_map,
        }
