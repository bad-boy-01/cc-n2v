"""
lib/agents/__init__.py — Base agent class for CC-Novel2Video free pipeline.

All agents share:
- Access to QwenClient (loaded on demand)
- Project path resolution via ProjectManager
- Logging + progress tracking
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from lib.config import DEFAULT_LLM
from lib.project_manager import ProjectManager

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Base class for all pipeline agents.

    Subclasses should override `run()`.
    The QwenClient is lazy-loaded on first use to support staged model loading.
    """

    def __init__(
        self,
        project_name: str,
        projects_root: Optional[str] = None,
        llm: str = DEFAULT_LLM,
        load_in_4bit: bool = True,
    ):
        self.project_name = project_name
        self.pm = ProjectManager(projects_root)
        self.project_dir = self.pm.get_project_path(project_name)
        self.llm_backend = llm
        self.load_in_4bit = load_in_4bit
        self._qwen: Any = None  # lazy

    @property
    def qwen(self):
        """Lazy-load QwenClient on first access."""
        if self._qwen is None:
            from lib.qwen_client import get_qwen_client
            self._qwen = get_qwen_client(
                llm=self.llm_backend,
                load_in_4bit=self.load_in_4bit,
            )
        return self._qwen

    def get_path(self, *parts: str) -> Path:
        """Return an absolute path inside the project directory."""
        return self.project_dir.joinpath(*parts)

    def ensure_dir(self, *parts: str) -> Path:
        """Create and return a directory inside the project."""
        p = self.get_path(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def run(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement run()")

    def log(self, msg: str) -> None:
        logger.info(f"[{self.__class__.__name__}][{self.project_name}] {msg}")
