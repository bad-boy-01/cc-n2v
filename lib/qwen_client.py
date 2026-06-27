"""
lib/qwen_client.py — Local LLM Client for CC-Novel2Video

Primary:   Qwen2.5-7B-Instruct     (thinking mode support)
Fallback:  Qwen2.5-7B-Instruct
Optional:  DeepSeek-R1-Distill-Qwen-7B  (better structured extraction)

Runs on Kaggle T4 via 4-bit bitsandbytes quantization.
Model is loaded once and cached in memory across all calls.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Model registry ────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "qwen2.5-7b": {
        "repo": "Qwen/Qwen2.5-7B-Instruct",
        "thinking": False,
        "context_length": 32768,
    },
    "deepseek-r1-7b": {
        "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "thinking": True,
        "context_length": 32768,
    },
}

DEFAULT_PRIMARY = "qwen2.5-7b"
DEFAULT_FALLBACK = "qwen2.5-7b"

# ── Thread-safe singleton cache ───────────────────────────────────────────────

_MODEL_LOCK = threading.Lock()
_LOADED_MODELS: Dict[str, Any] = {}   # model_key → {"model": ..., "tokenizer": ...}


def _load_model(model_key: str, load_in_4bit: bool = True) -> Dict[str, Any]:
    """Load a model + tokenizer, caching the result in memory."""
    with _MODEL_LOCK:
        if model_key in _LOADED_MODELS:
            return _LOADED_MODELS[model_key]

        config = MODEL_CONFIGS.get(model_key)
        if config is None:
            raise ValueError(f"Unknown model key: {model_key}. "
                             f"Choose from: {list(MODEL_CONFIGS)}")

        repo = config["repo"]
        logger.info(f"Loading model {repo} (4bit={load_in_4bit}) …")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch

            bnb_config = None
            if load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )

            tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                repo,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if not load_in_4bit else None,
            )
            model.eval()

            entry = {"model": model, "tokenizer": tokenizer, "config": config}
            _LOADED_MODELS[model_key] = entry
            logger.info(f"✅ Model {repo} loaded successfully.")
            return entry

        except Exception as e:
            logger.error(f"Failed to load {repo}: {e}")
            raise


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> reasoning blocks from Qwen2.5/DeepSeek-R1 output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> str:
    """Extract the first JSON object or array from a model response."""
    text = _strip_thinking(text)
    # Try to find JSON between ```json … ``` fences first
    fence_match = re.search(r"```(?:json)?\s*([\[\{].*?)\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    # Fallback: find first { or [ and take to matching close
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start != -1:
            depth, i = 0, start
            in_string = False
            escape = False
            while i < len(text):
                c = text[i]
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"' and not escape:
                    in_string = not in_string
                elif not in_string:
                    if c == start_char:
                        depth += 1
                    elif c == end_char:
                        depth -= 1
                        if depth == 0:
                            return text[start:i + 1]
                i += 1
    return text


class QwenClient:
    """
    Local LLM client wrapping Qwen2.5-7B-Instruct (primary)
    and Qwen2.5-7B-Instruct or DeepSeek-R1 (fallback/optional).

    Usage
    -----
    client = QwenClient()
    text   = client.generate("Summarize this story: ...")
    data   = client.generate_json("Extract characters from: ...")
    """

    def __init__(
        self,
        primary: str = DEFAULT_PRIMARY,
        fallback: str = DEFAULT_FALLBACK,
        load_in_4bit: bool = True,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        use_thinking: bool = False,
    ):
        self.primary_key = primary
        self.fallback_key = fallback
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_thinking = use_thinking
        self._active_key: Optional[str] = None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_entry(self, model_key: str) -> Dict[str, Any]:
        return _load_model(model_key, self.load_in_4bit)

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str],
        thinking: bool,
        model_key: str,
    ) -> list:
        config = MODEL_CONFIGS[model_key]
        messages = []

        sys_content = system_prompt or "You are a helpful AI assistant."
        if thinking and config.get("thinking"):
            sys_content += " /think"
        elif config.get("thinking"):
            sys_content += " /no_think"

        messages.append({"role": "system", "content": sys_content})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _run_inference(
        self,
        model_key: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        thinking: bool = False,
    ) -> str:
        t = temperature if temperature is not None else self.temperature
        n = max_new_tokens if max_new_tokens is not None else self.max_new_tokens

        # Support for Ollama Backend
        if os.environ.get("USE_OLLAMA") == "1":
            import requests
            ollama_model = "qwen2.5:7b"
            if "deepseek" in model_key:
                ollama_model = "deepseek-r1:7b"
            
            messages = self._build_messages(prompt, system_prompt, thinking, model_key)
            payload = {
                "model": ollama_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": t if t > 0 else 1.0,
                    "num_predict": n
                }
            }
            try:
                resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=600)
                resp.raise_for_status()
                return resp.json().get("message", {}).get("content", "")
            except Exception as e:
                logger.error(f"Ollama inference failed: {e}")
                raise

        import torch

        entry = self._get_entry(model_key)
        model = entry["model"]
        tokenizer = entry["tokenizer"]

        messages = self._build_messages(prompt, system_prompt, thinking, model_key)

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        gen_kwargs = {
            "max_new_tokens": n,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if t > 0:
            gen_kwargs["temperature"] = t
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_kwargs,
            )

        # Decode only the new tokens
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)

    def _try_models(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_new_tokens: Optional[int],
        thinking: bool,
    ) -> str:
        """Try primary, then fallback."""
        for key in [self.primary_key, self.fallback_key]:
            if key is None:
                continue
            try:
                result = self._run_inference(
                    key, prompt, system_prompt, temperature, max_new_tokens, thinking
                )
                self._active_key = key
                return result
            except Exception as e:
                logger.warning(f"Model {key} failed: {e}. Trying next …")
        raise RuntimeError("All LLM models failed. Check GPU memory and model files.")

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        thinking: bool = False,
    ) -> str:
        """
        Generate free-form text.

        Parameters
        ----------
        thinking : bool
            If True, enables Qwen2.5 chain-of-thought thinking mode.
            Thinking output is stripped from the returned string.
        """
        raw = self._try_models(
            prompt, system_prompt, temperature, max_new_tokens, thinking
        )
        return _strip_thinking(raw)

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_new_tokens: Optional[int] = None,
        thinking: bool = True,   # thinking helps structured extraction
    ) -> Dict[str, Any]:
        """
        Generate and parse a JSON object/array.

        The model is instructed to output valid JSON. Thinking blocks are
        stripped before parsing. Falls back to json-repair on malformed output.
        """
        json_sys = (system_prompt or "") + (
            "\n\nIMPORTANT: You must respond with valid JSON only. "
            "No explanation, no markdown fences, just the JSON object."
        )
        raw = self._try_models(
            prompt, json_sys, temperature, max_new_tokens, thinking
        )
        extracted = _extract_json(raw)

        # Try standard parse first
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

        # Try json-repair if available
        try:
            from json_repair import repair_json  # type: ignore
            repaired = repair_json(extracted)
            if isinstance(repaired, str):
                return json.loads(repaired)
            return repaired  # type: ignore
        except Exception:
            pass

        logger.warning(f"Could not parse JSON from model output. Raw: {extracted[:200]}")
        return {}

    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate JSON that conforms to a given schema dict.
        The schema is included in the system prompt for guidance.
        """
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        structured_sys = (
            (system_prompt or "You are a helpful AI assistant.")
            + f"\n\nOutput ONLY a valid JSON object matching this schema:\n{schema_str}"
        )
        return self.generate_json(
            prompt,
            system_prompt=structured_sys,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            thinking=True,
        )

    def unload_model(self) -> None:
        """
        Unload cached model(s) from GPU memory.
        Call this between pipeline stages to free VRAM before loading FLUX/Kokoro.

        Example (staged loading pattern):
            client = QwenClient()
            storyboard = client.generate_json(...)
            client.unload_model()          # ← free ~8 GB VRAM
            gc.collect()
            torch.cuda.empty_cache()
            # now load FLUX safely
        """
        import gc
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False

        with _MODEL_LOCK:
            keys_to_remove = list(_LOADED_MODELS.keys())
            for key in keys_to_remove:
                entry = _LOADED_MODELS.pop(key, None)
                if entry:
                    model = entry.get("model")
                    if model is not None:
                        try:
                            del model
                        except Exception:
                            pass
                    tokenizer = entry.get("tokenizer")
                    if tokenizer is not None:
                        try:
                            del tokenizer
                        except Exception:
                            pass
                logger.info(f"Unloaded model: {key}")

        gc.collect()
        if has_torch:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared after model unload.")

        self._active_key = None

    @property
    def active_model(self) -> Optional[str]:
        """Returns the key of the model used in the last call."""
        return self._active_key


# ── Specialised client for extraction tasks ───────────────────────────────────

class DeepSeekExtractionClient(QwenClient):
    """
    Uses DeepSeek-R1-Distill-Qwen-7B as primary for structured extraction.
    Falls back to Qwen2.5-7B automatically.

    Best for: character extraction, clue extraction, scene breakdown.
    Activate via --llm deepseek flag.
    """

    def __init__(self, **kwargs):
        super().__init__(
            primary="deepseek-r1-7b",
            fallback=DEFAULT_PRIMARY,
            **kwargs,
        )


# ── Singleton factory ─────────────────────────────────────────────────────────

_DEFAULT_CLIENT: Optional[QwenClient] = None
_CLIENT_LOCK = threading.Lock()


def get_qwen_client(
    llm: str = "qwen2.5-7b",
    load_in_4bit: bool = True,
) -> QwenClient:
    """
    Return (or create) the process-level singleton QwenClient.

    Parameters
    ----------
    llm : str
        "qwen2.5-7b"    → Qwen2.5-7B primary
        "deepseek" → DeepSeek-R1-7B primary (better extraction)
    """
    global _DEFAULT_CLIENT
    with _CLIENT_LOCK:
        if _DEFAULT_CLIENT is None:
            if llm == "deepseek":
                _DEFAULT_CLIENT = DeepSeekExtractionClient(load_in_4bit=load_in_4bit)
            else:
                _DEFAULT_CLIENT = QwenClient(load_in_4bit=load_in_4bit)
    return _DEFAULT_CLIENT
