"""
lib/prompt_optimizer.py — Image prompt cleaner and quality enhancer.

Takes a raw assembled prompt (from PromptBuilder) and:
  1. Deduplicates adjectives within a sliding window
  2. Removes contradictory term pairs
  3. Merges redundant synonyms
  4. Injects the quality suffix from DirectorProfile
  5. Trims to max token limit

No LLM required. Pure text processing.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Default quality suffix (overridden by DirectorProfile)
DEFAULT_QUALITY_SUFFIX = "masterpiece, best quality, highly detailed, 8k, sharp focus"

# Max output tokens (approximate — 1 token ≈ 4 chars)
DEFAULT_MAX_TOKENS = 400

# Window size for deduplication (words)
DEDUP_WINDOW = 8

# Contradictory term pairs — if both appear, remove the second one
CONTRADICTIONS: List[tuple] = [
    ("bright", "dark"),
    ("day", "night"),
    ("indoor", "outdoor"),
    ("close-up", "wide establishing"),
    ("wide shot", "close-up"),
    ("moonlit", "midday sun"),
    ("sunrise", "night"),
    ("sunset", "morning"),
]

# Synonym groups — keep only the first occurrence of any synonym
SYNONYM_GROUPS: List[List[str]] = [
    ["highly detailed", "detailed", "high detail", "intricate detail"],
    ["masterpiece", "best quality", "top quality", "highest quality"],
    ["8k", "8k resolution", "ultra high resolution", "uhd"],
    ["sharp focus", "sharp", "in focus"],
    ["dramatic lighting", "dramatic light", "dramatic illumination"],
    ["cinematic", "cinematic style", "cinematic look"],
    ["natural lighting", "natural light", "natural illumination"],
]

# Words to always strip from prompts (harmful or useless for FLUX/SDXL)
STRIP_WORDS: Set[str] = {
    "generate", "create", "draw", "make", "render", "show",
    "please", "can you", "could you", "i want", "give me",
    "image of", "picture of", "photo of",
}


class PromptOptimizer:
    """
    Image prompt cleaner and quality enhancer.

    Usage
    -----
    opt = PromptOptimizer(director_profile)
    clean = opt.optimize(raw_prompt)
    """

    def __init__(
        self,
        director_profile: Optional[Dict[str, Any]] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.profile = director_profile or {}
        self.max_tokens = max_tokens
        self._quality_suffix = self.profile.get(
            "quality_suffix", DEFAULT_QUALITY_SUFFIX
        )

    def optimize(self, raw_prompt: str) -> str:
        """
        Clean and enhance a raw image prompt.

        Steps:
        1. Strip harmful instruction words
        2. Normalize punctuation
        3. Deduplicate adjectives
        4. Remove contradictions
        5. Collapse synonym groups
        6. Inject quality suffix
        7. Trim to max tokens

        Returns
        -------
        str
            Cleaned, enhanced prompt ready for FLUX/SDXL.
        """
        if not raw_prompt or not raw_prompt.strip():
            return self._quality_suffix

        prompt = raw_prompt.strip()

        # Step 1: Strip harmful words
        prompt = self._strip_instruction_words(prompt)

        # Step 2: Normalize punctuation
        prompt = self._normalize_punctuation(prompt)

        # Step 3: Deduplicate repeated adjectives/phrases
        prompt = self._dedup_phrases(prompt)

        # Step 4: Remove contradicting terms
        prompt = self._remove_contradictions(prompt)

        # Step 5: Collapse synonym groups
        prompt = self._collapse_synonyms(prompt)

        # Step 6: Inject quality suffix
        prompt = self._inject_quality_suffix(prompt)

        # Step 7: Trim to token limit
        prompt = self._trim_to_limit(prompt)

        return prompt.strip()

    # ── Private methods ───────────────────────────────────────────────────────

    def _strip_instruction_words(self, prompt: str) -> str:
        """Remove words that confuse FLUX/SDXL diffusion models."""
        lower = prompt.lower()
        for word in STRIP_WORDS:
            if word in lower:
                # Case-insensitive replace
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                prompt = pattern.sub("", prompt)
        return prompt

    def _normalize_punctuation(self, prompt: str) -> str:
        """Normalize multiple punctuation marks and whitespace."""
        # Remove multiple consecutive commas
        prompt = re.sub(r",\s*,+", ",", prompt)
        # Remove multiple consecutive periods
        prompt = re.sub(r"\.\s*\.+", ".", prompt)
        # Normalize whitespace
        prompt = re.sub(r"\s+", " ", prompt)
        # Remove leading/trailing punctuation
        prompt = prompt.strip("., ")
        return prompt

    def _dedup_phrases(self, prompt: str) -> str:
        """
        Remove duplicate words within a sliding window.

        This catches cases like "dark dark forest" or "dramatic dramatic lighting".
        Only removes exact word duplicates, not semantic duplicates.
        """
        # Split on word boundaries, preserving delimiters
        tokens = re.split(r"(\s+|,\s*|\.\s*)", prompt)
        words = [t for t in tokens if t.strip() and not t.strip() in (",", ".")]
        separators = [t for t in tokens if not t.strip() or t.strip() in (",", ".")]

        seen_in_window: List[str] = []
        result_words = []

        for word in words:
            word_lower = word.lower().strip()
            if word_lower and len(word_lower) > 3:  # skip short words
                if word_lower in seen_in_window:
                    continue  # duplicate within window, skip
                seen_in_window.append(word_lower)
                if len(seen_in_window) > DEDUP_WINDOW:
                    seen_in_window.pop(0)
            result_words.append(word)

        # Reconstruct (simplified)
        return " ".join(result_words)

    def _remove_contradictions(self, prompt: str) -> str:
        """Remove the second word when two contradicting terms appear."""
        lower = prompt.lower()
        for term_a, term_b in CONTRADICTIONS:
            if term_a in lower and term_b in lower:
                # Remove term_b (less important, appears second)
                pattern = re.compile(re.escape(term_b), re.IGNORECASE)
                prompt = pattern.sub("", prompt)
                lower = prompt.lower()
        return prompt

    def _collapse_synonyms(self, prompt: str) -> str:
        """Keep only the first synonym from each synonym group."""
        lower = prompt.lower()
        for group in SYNONYM_GROUPS:
            found_first = False
            for syn in group:
                if syn in lower:
                    if found_first:
                        # Remove subsequent synonyms
                        pattern = re.compile(re.escape(syn), re.IGNORECASE)
                        prompt = pattern.sub("", prompt)
                        lower = prompt.lower()
                    else:
                        found_first = True
        return prompt

    def _inject_quality_suffix(self, prompt: str) -> str:
        """Append quality suffix if not already present."""
        suffix_words = set(self._quality_suffix.lower().split(","))
        prompt_lower = prompt.lower()

        # Only inject if suffix words are not already in prompt
        missing = [
            w.strip() for w in suffix_words
            if w.strip() and w.strip() not in prompt_lower
        ]

        if missing:
            prompt = prompt.rstrip("., ") + ", " + self._quality_suffix
        return prompt

    def _trim_to_limit(self, prompt: str) -> str:
        """Trim prompt to approximate token limit."""
        # Approximate: 1 token ≈ 4 characters
        max_chars = self.max_tokens * 4
        if len(prompt) <= max_chars:
            return prompt

        # Trim at the last comma before the limit
        trimmed = prompt[:max_chars]
        last_comma = trimmed.rfind(",")
        if last_comma > max_chars * 0.8:
            trimmed = trimmed[:last_comma]

        # Always keep the quality suffix at the end
        suffix = ", " + self._quality_suffix
        if not trimmed.endswith(self._quality_suffix):
            trim_limit = max_chars - len(suffix)
            trimmed = trimmed[:trim_limit].rstrip("., ")
            trimmed += suffix

        return trimmed
