"""plugins/llm/qwen.py — QwenClient LLM plugin wrapper."""
from __future__ import annotations
from typing import Any
from plugins.base import LLMPlugin


class Plugin(LLMPlugin):
    def __init__(self, model_id: str = "qwen2.5-7b"):
        self.model_id = model_id
        self._client = None

    def load(self, load_in_4bit: bool = True) -> None:
        from lib.qwen_client import get_qwen_client
        self._client = get_qwen_client(llm=self.model_id, load_in_4bit=load_in_4bit)

    def generate(
        self, prompt, system_prompt="", temperature=0.7, max_tokens=2048, thinking=False
    ) -> str:
        self._ensure_loaded()
        return self._client.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_new_tokens=max_tokens,
            thinking=thinking,
        )

    def generate_json(self, prompt, temperature=0.1, thinking=False) -> Any:
        self._ensure_loaded()
        return self._client.generate_json(
            prompt, temperature=temperature, thinking=thinking
        )

    def unload(self) -> None:
        if self._client:
            try:
                self._client.unload_model()
            except Exception:
                pass
            self._client = None

    def _ensure_loaded(self):
        if self._client is None:
            self.load()

    @property
    def plugin_id(self) -> str:
        return self.model_id

    @property
    def vram_gb(self) -> float:
        return 5.0
