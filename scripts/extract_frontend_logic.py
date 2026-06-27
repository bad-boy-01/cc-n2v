#!/usr/bin/env python3
"""
scripts/extract_frontend_logic.py — Frontend logic audit tool.

Scans the CC-Novel2Video codebase for:
  - Business logic hidden inside UI/web layers
  - Hardcoded prompt templates
  - Model configuration buried in frontend files
  - API call patterns that reveal workflows
  - Generation settings in JS/HTML files

Produces a migration report showing what needs to move into lib/services/.

Usage:
    python scripts/extract_frontend_logic.py
    python scripts/extract_frontend_logic.py --output report.md
    python scripts/extract_frontend_logic.py --path /path/to/project
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Patterns to search for ────────────────────────────────────────────────────

# Python: model/API calls that shouldn't be in UI files
PYTHON_API_PATTERNS = [
    (r"GeminiClient\s*\(", "Direct GeminiClient instantiation"),
    (r"GeminiClient\b", "GeminiClient reference"),
    (r"gemini_client\b", "gemini_client reference"),
    (r"generate_image\s*\(", "generate_image() call"),
    (r"generate_video\s*\(", "generate_video() call"),
    (r"MediaGenerator\s*\(", "MediaGenerator instantiation"),
    (r"rate_limiter\b", "rate_limiter reference"),
    (r"ffmpeg", "FFmpeg call"),
    (r"subprocess\.run", "subprocess.run (possible ffmpeg/model call)"),
    (r"torch\.", "PyTorch call"),
    (r"diffusers\b", "diffusers import"),
    (r"transformers\b", "transformers import"),
    (r"prompt_builders\b", "prompt_builders import"),
    (r"build_character_prompt\b", "build_character_prompt() call"),
    (r"build_clue_prompt\b", "build_clue_prompt() call"),
    (r"image_prompt_to_yaml\b", "image_prompt_to_yaml() call"),
    (r"video_prompt_to_yaml\b", "video_prompt_to_yaml() call"),
    (r"normalize_veo_duration", "Veo duration normalization logic"),
    (r"_build_grid_prompt", "_build_grid_prompt() — grid storyboard logic"),
    (r"_collect_reference_images", "_collect_reference_images() — reference logic"),
    (r"execute_storyboard_task", "execute_storyboard_task() — task logic"),
    (r"execute_video_task", "execute_video_task() — task logic"),
    (r"execute_character_task", "execute_character_task() — task logic"),
    (r"execute_clue_task", "execute_clue_task() — task logic"),
]

# JavaScript: business logic that shouldn't be in frontend files
JS_LOGIC_PATTERNS = [
    (r"prompt\s*=\s*[`'\"].*[`'\"]", "Hardcoded prompt string"),
    (r"temperature\s*[:=]\s*[\d.]+", "Hardcoded temperature"),
    (r"model\s*[:=]\s*['\"].*['\"]", "Hardcoded model name"),
    (r"guidance_scale", "guidance_scale parameter"),
    (r"num_inference_steps", "num_inference_steps parameter"),
    (r"max_tokens", "max_tokens parameter"),
    (r"aspect_ratio\s*[:=]\s*['\"]", "Hardcoded aspect ratio"),
    (r"duration_seconds\s*[:=]\s*\d+", "Hardcoded duration"),
    (r"ffmpeg", "FFmpeg reference in JS"),
    (r"fetch\s*\(\s*['\"](?!\/api)", "Non-API fetch call (possible external service)"),
]

# HTML: embedded logic or settings
HTML_LOGIC_PATTERNS = [
    (r"<script>.*?(model|prompt|temperature|ffmpeg).*?</script>", "Inline script with model/prompt settings"),
    (r"data-model\s*=", "data-model attribute (hardcoded model setting)"),
    (r"data-prompt\s*=", "data-prompt attribute (hardcoded prompt)"),
]

# Prompt templates — strings that look like generation prompts
PROMPT_TEMPLATE_PATTERNS = [
    r"一张.*分镜图",              # "a storyboard image" (Chinese)
    r"画面.*描述",                 # "scene description"
    r"人物.*与.*参考图.*一致",     # "character must match reference"
    r"generate.*image.*of",
    r"anime.*illustration",
    r"cinematic.*shot",
    r"photorealistic",
    r"highly detailed",
    r"style.*:.*\{.*\}",
]


# ── Scanner ───────────────────────────────────────────────────────────────────

class Finding:
    def __init__(self, file: Path, line: int, pattern: str, snippet: str, severity: str = "medium"):
        self.file = file
        self.line = line
        self.pattern = pattern
        self.snippet = snippet.strip()[:120]
        self.severity = severity  # "high", "medium", "low"

    def to_dict(self) -> Dict:
        return {
            "file": str(self.file),
            "line": self.line,
            "pattern": self.pattern,
            "snippet": self.snippet,
            "severity": self.severity,
        }


class FrontendAuditor:
    """Scans the codebase for business logic trapped in UI layers."""

    def __init__(self, project_root: Path):
        self.root = project_root
        self.findings: List[Finding] = []

    # ── UI file identification ─────────────────────────────────────────────

    def _is_ui_python_file(self, path: Path) -> bool:
        """Return True if this Python file is in a UI/web layer."""
        parts = {p.lower() for p in path.parts}
        ui_dirs = {"webui", "server", "routers", "frontend", "views", "templates"}
        return bool(parts & ui_dirs)

    def _is_business_logic_python(self, path: Path) -> bool:
        """Return True if this Python file is supposed to contain business logic."""
        # These are the web services that call models directly — key audit targets
        name = path.name.lower()
        return name in {
            "generation_tasks.py",
            "generate.py",
            "projects.py",
            "characters.py",
            "clues.py",
            "tasks.py",
        }

    # ── Scanners ──────────────────────────────────────────────────────────

    def _scan_python_file(self, path: Path) -> None:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
        except Exception:
            return

        is_ui = self._is_ui_python_file(path)
        is_service = self._is_business_logic_python(path)

        for i, line in enumerate(lines, 1):
            for pattern, description in PYTHON_API_PATTERNS:
                if re.search(pattern, line):
                    severity = "high" if is_ui or is_service else "low"
                    self.findings.append(Finding(
                        file=path.relative_to(self.root),
                        line=i,
                        pattern=description,
                        snippet=line,
                        severity=severity,
                    ))

            # Detect embedded prompt templates
            for pt in PROMPT_TEMPLATE_PATTERNS:
                if re.search(pt, line, re.IGNORECASE):
                    self.findings.append(Finding(
                        file=path.relative_to(self.root),
                        line=i,
                        pattern="Embedded prompt template",
                        snippet=line,
                        severity="medium" if is_ui else "low",
                    ))

    def _scan_js_file(self, path: Path) -> None:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
        except Exception:
            return

        for i, line in enumerate(lines, 1):
            for pattern, description in JS_LOGIC_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    self.findings.append(Finding(
                        file=path.relative_to(self.root),
                        line=i,
                        pattern=description,
                        snippet=line,
                        severity="medium",
                    ))

    def _scan_html_file(self, path: Path) -> None:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return

        for pattern, description in HTML_LOGIC_PATTERNS:
            for m in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
                line = content[: m.start()].count("\n") + 1
                self.findings.append(Finding(
                    file=path.relative_to(self.root),
                    line=line,
                    pattern=description,
                    snippet=m.group()[:120],
                    severity="low",
                ))

    def scan(self) -> List[Finding]:
        """Scan entire project and return all findings."""
        self.findings = []

        # Python files — focus on webui/
        for py_file in self.root.rglob("*.py"):
            if any(p in str(py_file) for p in [".venv", "node_modules", "__pycache__", ".git"]):
                continue
            self._scan_python_file(py_file)

        # JavaScript files
        for js_file in self.root.rglob("*.js"):
            if any(p in str(js_file) for p in ["node_modules", ".git"]):
                continue
            self._scan_js_file(js_file)

        # HTML files
        for html_file in self.root.rglob("*.html"):
            self._scan_html_file(html_file)

        return self.findings

    # ── Report generation ─────────────────────────────────────────────────

    def generate_report(self) -> str:
        findings = self.findings
        high = [f for f in findings if f.severity == "high"]
        medium = [f for f in findings if f.severity == "medium"]
        low = [f for f in findings if f.severity == "low"]

        lines = [
            "# CC-Novel2Video Frontend Logic Audit Report",
            "",
            "## Summary",
            "",
            f"| Severity | Count |",
            f"|---|---|",
            f"| 🔴 High (business logic in UI layer) | {len(high)} |",
            f"| 🟡 Medium (logic that should move to services) | {len(medium)} |",
            f"| 🟢 Low (minor references) | {len(low)} |",
            f"| **Total** | **{len(findings)}** |",
            "",
            "---",
            "",
            "## Migration Priority",
            "",
            "### 🔴 High Priority — Move to `lib/services/` immediately",
            "",
        ]

        if high:
            # Group by file
            by_file: Dict[str, List[Finding]] = {}
            for f in high:
                key = str(f.file)
                by_file.setdefault(key, []).append(f)

            for filepath, file_findings in sorted(by_file.items()):
                lines.append(f"#### `{filepath}`")
                for f in file_findings:
                    lines.append(f"- **Line {f.line}** — {f.pattern}")
                    lines.append(f"  ```")
                    lines.append(f"  {f.snippet}")
                    lines.append(f"  ```")
                lines.append("")
        else:
            lines.append("_No high-severity findings._\n")

        lines += [
            "### 🟡 Medium Priority — Extract to service functions",
            "",
        ]

        if medium:
            by_file = {}
            for f in medium:
                key = str(f.file)
                by_file.setdefault(key, []).append(f)

            for filepath, file_findings in sorted(by_file.items()):
                lines.append(f"#### `{filepath}`")
                for f in file_findings:
                    lines.append(f"- **Line {f.line}** — {f.pattern}")
                lines.append("")
        else:
            lines.append("_No medium-severity findings._\n")

        lines += [
            "---",
            "",
            "## Recommended Service Mapping",
            "",
            "| Current Location | Move To |",
            "|---|---|",
            "| `webui/server/services/generation_tasks.py` → `execute_storyboard_task()` | `lib/services/storyboard_service.py` |",
            "| `webui/server/services/generation_tasks.py` → `execute_video_task()` | `lib/services/video_service.py` |",
            "| `webui/server/services/generation_tasks.py` → `execute_character_task()` | `lib/services/image_service.py` |",
            "| `webui/server/services/generation_tasks.py` → `execute_clue_task()` | `lib/services/image_service.py` |",
            "| `webui/server/services/generation_tasks.py` → `_build_grid_prompt()` | `lib/services/storyboard_service.py` |",
            "| `webui/server/services/generation_tasks.py` → `_collect_reference_images()` | `lib/services/storyboard_service.py` |",
            "| `webui/server/services/generation_tasks.py` → `_normalize_storyboard_prompt()` | `lib/services/storyboard_service.py` |",
            "| `lib/gemini_client.py` (all image generation) | `lib/services/image_service.py` → `lib/image_generator.py` |",
            "| Veo video generation | DELETE — replaced by `lib/motion_engine.py` |",
            "",
            "---",
            "",
            "## Target Architecture",
            "",
            "```",
            "WebUI (thin layer)              CLI / Notebook",
            "     |                               |",
            "     └─────────────┬─────────────────┘",
            "                   |",
            "            lib/services/",
            "         ┌─────────┴──────────┐",
            "         |                    |",
            "  storyboard_service   pipeline_service",
            "  image_service        audio_service",
            "  video_service        subtitle_service",
            "         |",
            "   lib/agents/ + lib/image_generator.py",
            "   lib/motion_engine.py + lib/kokoro_tts.py",
            "```",
            "",
            "After migration: No model calls, FFmpeg calls, or OCR calls inside `webui/`.",
            "",
        ]

        return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Audit CC-Novel2Video frontend for trapped business logic")
    parser.add_argument("--path", default=".", help="Project root directory")
    parser.add_argument("--output", default=None, help="Write report to this file (default: print to stdout)")
    parser.add_argument("--json", action="store_true", help="Output raw findings as JSON")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    if not root.exists():
        print(f"Error: path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {root} …", file=sys.stderr)
    auditor = FrontendAuditor(root)
    findings = auditor.scan()
    print(f"Found {len(findings)} findings.", file=sys.stderr)

    if args.json:
        output = json.dumps([f.to_dict() for f in findings], indent=2, ensure_ascii=False)
    else:
        output = auditor.generate_report()

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
