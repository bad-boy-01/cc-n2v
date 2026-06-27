"""
lib/prompt_validator.py — Final prompt validation before FLUX/SDXL.

Validates the optimized prompt and returns a ValidationResult with
warnings and an optionally auto-corrected prompt.

Checks
------
1. Minimum length (prompt too short)
2. Duplicate 3+ word phrases
3. Character names present (if scene has characters)
4. Location keyword present (if scene has location)
5. Contradictory lighting terms
6. Contradictory camera terms
7. Token limit (hard max)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Validation thresholds
MIN_PROMPT_LENGTH = 30       # characters
MAX_PROMPT_TOKENS = 450      # approximate tokens
MAX_CHARS = MAX_PROMPT_TOKENS * 4

# Contradictory lighting pairs
LIGHTING_CONTRADICTIONS = [
    ("bright moonlight", "midday sun"),
    ("dark night", "bright daylight"),
    ("moonlit", "bright sunny"),
    ("sunset", "moonlit night"),
    ("dawn", "nighttime"),
]

# Contradictory camera pairs
CAMERA_CONTRADICTIONS = [
    ("close-up", "wide establishing"),
    ("close-up portrait", "wide shot"),
    ("extreme close-up", "wide angle"),
]

# Common NSFW / model-breaking terms to flag
BLOCKED_TERMS = [
    "nsfw", "nude", "naked", "explicit",
    "generate an image", "create an image", "draw me",
]


@dataclass
class ValidationResult:
    """Result of prompt validation."""
    ok: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    fixed_prompt: str = ""
    original_prompt: str = ""

    def has_issues(self) -> bool:
        return bool(self.warnings or self.errors)

    def summary(self) -> str:
        status = "✅ PASS" if self.ok else "❌ FAIL"
        lines = [f"{status} — {len(self.warnings)} warnings, {len(self.errors)} errors"]
        for w in self.warnings:
            lines.append(f"  ⚠️  {w}")
        for e in self.errors:
            lines.append(f"  ❌ {e}")
        return "\n".join(lines)


class PromptValidator:
    """
    Final prompt validator before sending to FLUX/SDXL.

    Returns a ValidationResult with ok=True if the prompt passes
    all checks, plus a `fixed_prompt` that has auto-corrections applied.

    Usage
    -----
    validator = PromptValidator()
    result = validator.validate(prompt, scene)

    if not result.ok:
        print(result.summary())
        prompt = result.fixed_prompt  # use auto-fixed version
    """

    def __init__(
        self,
        min_length: int = MIN_PROMPT_LENGTH,
        max_tokens: int = MAX_PROMPT_TOKENS,
        strict: bool = False,
    ):
        """
        Parameters
        ----------
        min_length : int
            Minimum character length for a valid prompt
        max_tokens : int
            Maximum approximate token count
        strict : bool
            If True, warnings become errors (fail-fast mode)
        """
        self.min_length = min_length
        self.max_tokens = max_tokens
        self.strict = strict

    def validate(
        self,
        prompt: str,
        scene: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate a prompt against the scene metadata.

        Parameters
        ----------
        prompt : str
            The optimized prompt to validate
        scene : dict, optional
            Scene metadata for character/location cross-referencing

        Returns
        -------
        ValidationResult
            ok=True if prompt passes all checks.
            fixed_prompt contains auto-corrected prompt.
        """
        warnings: List[str] = []
        errors: List[str] = []
        fixed = prompt

        scene = scene or {}

        # ── 1. Minimum length ─────────────────────────────────────────────────
        if len(prompt.strip()) < self.min_length:
            errors.append(
                f"Prompt too short ({len(prompt.strip())} chars, min {self.min_length}). "
                f"Scene: {scene.get('segment_id', '?')}"
            )
            # Auto-fix: append scene description
            fallback = scene.get("novel_text", "")[:200]
            if fallback:
                fixed = (fixed + ". " + fallback).strip()

        # ── 2. Duplicate 3+ word phrases ─────────────────────────────────────
        dup_phrases = self._find_duplicate_phrases(prompt, min_words=3)
        if dup_phrases:
            warnings.append(f"Duplicate phrases: {dup_phrases[:3]}")
            for phrase in dup_phrases:
                fixed = self._remove_second_occurrence(fixed, phrase)

        # ── 3. Character names present ────────────────────────────────────────
        char_names = scene.get("characters", [])
        if char_names:
            missing_chars = [
                name for name in char_names
                if name.lower().split()[0] not in prompt.lower()  # check first name
            ]
            if missing_chars:
                warnings.append(
                    f"Character(s) not referenced in prompt: {missing_chars}. "
                    f"Consider adding character descriptions."
                )
                # Auto-fix: we can't add character descriptions here (no WorldEngine)
                # Just log the warning

        # ── 4. Location keyword present ───────────────────────────────────────
        location = scene.get("location", "")
        if location and len(location) > 3:
            # Check if first meaningful word of location is in prompt
            loc_words = [w for w in location.lower().split() if len(w) > 3]
            if loc_words and not any(w in prompt.lower() for w in loc_words):
                warnings.append(
                    f"Location '{location}' not referenced in prompt."
                )
                # Auto-fix: prepend location
                fixed = f"In {location}, {fixed}"

        # ── 5. Contradictory lighting ─────────────────────────────────────────
        prompt_lower = prompt.lower()
        for term_a, term_b in LIGHTING_CONTRADICTIONS:
            if term_a in prompt_lower and term_b in prompt_lower:
                warnings.append(f"Contradictory lighting: '{term_a}' vs '{term_b}'")
                # Auto-fix: remove term_b
                fixed = re.sub(re.escape(term_b), "", fixed, flags=re.IGNORECASE)

        # ── 6. Contradictory camera ───────────────────────────────────────────
        for term_a, term_b in CAMERA_CONTRADICTIONS:
            if term_a in prompt_lower and term_b in prompt_lower:
                warnings.append(f"Contradictory camera: '{term_a}' vs '{term_b}'")
                fixed = re.sub(re.escape(term_b), "", fixed, flags=re.IGNORECASE)

        # ── 7. Blocked terms ──────────────────────────────────────────────────
        for term in BLOCKED_TERMS:
            if term in prompt_lower:
                errors.append(f"Blocked term found: '{term}'")
                fixed = re.sub(re.escape(term), "", fixed, flags=re.IGNORECASE)

        # ── 8. Token limit ────────────────────────────────────────────────────
        approx_tokens = len(prompt) // 4
        if approx_tokens > self.max_tokens:
            warnings.append(
                f"Prompt too long (~{approx_tokens} tokens, max {self.max_tokens}). "
                f"Will be trimmed."
            )
            max_chars = self.max_tokens * 4
            fixed = fixed[:max_chars].rsplit(",", 1)[0]

        # ── Clean up fixed prompt ─────────────────────────────────────────────
        fixed = re.sub(r"\s+", " ", fixed).strip(". ,")

        # ── Determine ok ─────────────────────────────────────────────────────
        has_errors = bool(errors)
        has_warnings = bool(warnings)

        if self.strict:
            ok = not has_errors and not has_warnings
        else:
            ok = not has_errors

        return ValidationResult(
            ok=ok,
            warnings=warnings,
            errors=errors,
            fixed_prompt=fixed if fixed else prompt,
            original_prompt=prompt,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_duplicate_phrases(
        self, prompt: str, min_words: int = 3
    ) -> List[str]:
        """Find repeated multi-word phrases in the prompt."""
        words = prompt.lower().split()
        duplicates = []

        for n in range(min_words, min(8, len(words) // 2 + 1)):
            ngrams: Dict[str, int] = {}
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i: i + n])
                ngrams[phrase] = ngrams.get(phrase, 0) + 1
            for phrase, count in ngrams.items():
                if count > 1 and phrase not in duplicates:
                    duplicates.append(phrase)

        return duplicates

    def _remove_second_occurrence(self, text: str, phrase: str) -> str:
        """Remove the second occurrence of a phrase in text."""
        lower = text.lower()
        first_pos = lower.find(phrase.lower())
        if first_pos == -1:
            return text
        second_pos = lower.find(phrase.lower(), first_pos + 1)
        if second_pos == -1:
            return text
        return text[:second_pos] + text[second_pos + len(phrase):]
