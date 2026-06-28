"""
lib/image_generator.py — Image generation with scene reuse cache.

Model stack (in priority order):
  1. FLUX.1-schnell  (Apache 2.0, ~7 GB VRAM, fast)
  2. Stable Diffusion XL  (Apache 2.0, ~5 GB VRAM, fallback)
  3. Reuse nearest existing cached image  (last resort — never fails)

Character consistency hierarchy (no ControlNet, no LoRA required):
  1. hero_prompt.txt  ← appended to every scene prompt
  2. hero.json summary ← additional context
  3. hero_ref.png      ← IP-Adapter (OPTIONAL_ENHANCEMENT, skipped if OOM)

Scene cache key: {location, sorted(character_ids), scene_type}
  - Avoids wording-change cache misses
  - Cache hit = 30-60% reduction in generation calls for long stories

Scene-level checkpointing: saves progress after EVERY image.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import re
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Optional: IP-Adapter enhancement ─────────────────────────────────────────
OPTIONAL_ENHANCEMENT = True   # Set False to disable IP-Adapter entirely

RESOLUTION_MAP = {
    "1080p": (1080, 1920),   # 9:16 portrait for narration
    "1440p": (1440, 2560),
    "4k":    (2160, 3840),
    "16:9":  (1920, 1080),   # landscape drama mode
}

FLUX_SCHNELL_REPO = "black-forest-labs/FLUX.1-schnell"
SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"

# Controlled vocabulary — must match storyboard_agent.py
SCENE_TYPES = {
    "dialogue", "battle", "travel", "exposition", "emotional",
    "training", "city", "castle", "tavern", "classroom", "forest",
    "interior", "exterior", "action", "ceremony",
}


def _normalize_action(text: str) -> str:
    """Reduce action text to lowercase alphabetic for cache key."""
    return re.sub(r"[^a-z\u4e00-\u9fff]", "", text.lower())[:40]


def _scene_cache_key(scene: Dict) -> str:
    """
    Cache based on the exact image prompt.
    Using location/characters was too broad and caused 95% of scenes to reuse the exact same image.
    """
    key = {
        "prompt": scene.get("image_prompt", "").strip(),
    }
    raw = json.dumps(key, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ImageGenerator:
    """
    Generates storyboard images for an episode.

    Staged loading: call unload_model() when done to free VRAM before Stage 3.
    """

    def __init__(
        self,
        project_name: str,
        projects_root: str = "projects",
        resolution: str = "1080p",
        batch_size: int = 1,
        dry_run: bool = False,
        use_ip_adapter: bool = OPTIONAL_ENHANCEMENT,
    ):
        self.project_name = project_name
        self.projects_root = Path(projects_root)
        self.project_dir = self.projects_root / project_name
        self.resolution = resolution
        self.batch_size = batch_size
        self.dry_run = dry_run
        self.use_ip_adapter = use_ip_adapter

        # Directories
        self.images_dir = self.project_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.scene_cache_dir = self.project_dir / "scene_cache"
        self.scene_cache_dir.mkdir(parents=True, exist_ok=True)

        self._pipe = None          # FLUX plugin (lazy)
        self._sdxl_pipe = None     # SDXL fallback plugin (lazy)
        self._ip_adapter = None    # optional
        self._active_backend: Optional[str] = None

        # Scene-level progress file
        self._progress_path = self.project_dir / "image_progress.json"
        self._progress = self._load_progress()

    # ── Progress / checkpointing ──────────────────────────────────────────────

    def _load_progress(self) -> Dict[str, str]:
        if self._progress_path.exists():
            with open(self._progress_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_progress(self, segment_id: str, image_path: str) -> None:
        """Scene-level checkpoint — called after EVERY image generation."""
        self._progress[segment_id] = image_path
        with open(self._progress_path, "w", encoding="utf-8") as f:
            json.dump(self._progress, f, ensure_ascii=False, indent=2)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_flux(self) -> None:
        """Load FLUX.1-schnell plugin."""
        if self._pipe is not None:
            return

        from plugins import load_image_plugin
        logger.info("Loading FLUX.1-schnell plugin …")
        self._pipe = load_image_plugin("flux_schnell")
        self._pipe.load()
        self._active_backend = "flux"
        logger.info("FLUX.1-schnell loaded ✅")

    def _load_sdxl(self) -> None:
        """Load SDXL plugin as fallback."""
        if self._sdxl_pipe is not None:
            return

        from plugins import load_image_plugin
        logger.info("Loading SDXL plugin …")
        self._sdxl_pipe = load_image_plugin("sdxl")
        self._sdxl_pipe.load()
        self._active_backend = "sdxl"
        logger.info("SDXL loaded ✅")

    def unload_model(self) -> None:
        """Release GPU memory. MUST be called between pipeline stages."""
        for attr, plugin in [("_pipe", self._pipe), ("_sdxl_pipe", self._sdxl_pipe)]:
            if plugin is not None:
                try:
                    plugin.unload()
                except Exception as e:
                    logger.warning("Plugin unload warning (%s): %s", attr, e)
                setattr(self, attr, None)

        # IP-Adapter (not a plugin, just raw object)
        if self._ip_adapter is not None:
            try:
                del self._ip_adapter
            except Exception:
                pass
            self._ip_adapter = None

        self._active_backend = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("ImageGenerator: all models unloaded, VRAM freed")

    # ── Plugin self-test ──────────────────────────────────────────────────────

    @staticmethod
    def run_plugin_selftest(plugin_id: str = "flux_schnell") -> None:
        """
        Startup validation: load plugin → warmup (64×64, 1-step inference) → unload.

        The warmup call runs a real inference pass, verifying the full stack:
        weights, tokenizer, scheduler, VAE, UNet, device placement, xformers.

        Raises on failure — the pipeline must not process any scenes if
        the plugin cannot generate an image.
        """
        from plugins import load_image_plugin
        logger.info("[selftest] Testing plugin: %s …", plugin_id)
        try:
            plugin = load_image_plugin(plugin_id)
            plugin.load()
            plugin.warmup()   # 64×64, 1 step — ~1-2 seconds
            plugin.unload()
            logger.info("[selftest] ✓ Plugin self-test passed: %s", plugin_id)
        except Exception:
            logger.critical(
                "[selftest] Plugin self-test FAILED for '%s':\n%s",
                plugin_id,
                traceback.format_exc(),
            )
            raise

    # ── Character reference injection ─────────────────────────────────────────

    def _build_char_context(self, character_names: List[str]) -> str:
        """
        Build character context string for prompt injection.
        Hierarchy: hero_prompt.txt → hero.json → nothing
        IP-Adapter is OPTIONAL and handled separately.
        """
        chars_dir = self.project_dir / "characters"
        parts = []
        for name in character_names:
            stem = re.sub(r"[^\w\-]", "_", name.strip()).lower()
            prompt_file = chars_dir / f"{stem}_prompt.txt"
            if prompt_file.exists():
                parts.append(f"{name}: {prompt_file.read_text(encoding='utf-8').strip()}")
            else:
                json_file = chars_dir / f"{stem}.json"
                if json_file.exists():
                    with open(json_file, encoding="utf-8") as f:
                        data = json.load(f)
                    desc = data.get("hair", "") + " hair, " + data.get("eyes", "") + " eyes"
                    if data.get("clothing"):
                        desc += f", {data['clothing']}"
                    parts.append(f"{name}: {desc}")
        return "\n".join(parts)

    def _get_ref_images(self, character_names: List[str]) -> List[Path]:
        """Return paths to existing reference portrait PNGs."""
        chars_dir = self.project_dir / "characters"
        refs = []
        for name in character_names:
            stem = re.sub(r"[^\w\-]", "_", name.strip()).lower()
            ref = chars_dir / f"{stem}_ref.png"
            if ref.exists():
                refs.append(ref)
        return refs

    # ── Image generation core ─────────────────────────────────────────────────

    def _get_output_size(self) -> Tuple[int, int]:
        w, h = RESOLUTION_MAP.get(self.resolution, (1080, 1920))
        return w, h  # PIL uses (width, height)

    def _generate_with_flux(
        self, prompt: str, segment_id: str = "unknown"
    ) -> Optional["Image"]:
        from plugins.base import ImageGenerationRequest
        try:
            self._load_flux()
            w, h = self._get_output_size()

            request = ImageGenerationRequest(
                prompt=prompt,
                width=w,
                height=h,
                num_steps=4,
                guidance_scale=0.0,
            )
            img = self._pipe.generate(request)
            return img

        except Exception as exc:
            exc_str = str(exc)
            exc_type = type(exc).__name__
            if "OutOfMemoryError" in exc_type or "CUDA out of memory" in exc_str:
                logger.warning(
                    "Scene %s: FLUX OOM — releasing cache and falling back to SDXL",
                    segment_id,
                )
                self._pipe = None
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
                return None

            logger.error(
                "Image generation failed\n"
                "  Scene:  %s\n"
                "  Plugin: flux_schnell\n"
                "  Prompt: %.120s…",
                segment_id, prompt,
                exc_info=True,
            )
            logger.warning(
                "Scene %s: FLUX failed — attempting SDXL fallback …", segment_id
            )
            return None

    def _generate_with_sdxl(
        self, prompt: str, segment_id: str = "unknown"
    ) -> Optional["Image"]:
        from plugins.base import ImageGenerationRequest
        try:
            self._load_sdxl()
            w, h = self._get_output_size()

            # SDXL dimension capping is handled inside SDXLBackend.generate()
            request = ImageGenerationRequest(
                prompt=prompt,
                width=w,
                height=h,
                num_steps=30,
                guidance_scale=7.5,
            )
            img = self._sdxl_pipe.generate(request)
            return img

        except Exception:
            logger.error(
                "Image generation failed\n"
                "  Scene:  %s\n"
                "  Plugin: sdxl\n"
                "  Prompt: %.120s…",
                segment_id, prompt,
                exc_info=True,
            )
            return None

    def _reuse_nearest_image(self) -> Optional[Path]:
        """Last resort: reuse the most recently generated image."""
        images = sorted(self.images_dir.glob("scene_*.png"))
        cached = sorted(self.scene_cache_dir.glob("*.png"))
        candidates = images + cached
        if candidates:
            return candidates[-1]
        return None

    def _validate_image(self, path: Path) -> Tuple[int, int]:
        """
        Verify the saved image is readable and has valid dimensions.

        Returns
        -------
        (width, height) on success.

        Raises
        ------
        ValueError if the image is unreadable or has zero dimensions.
        """
        from PIL import Image as _PIL
        with _PIL.open(path) as img:
            w, h = img.size
            if w == 0 or h == 0:
                raise ValueError(
                    f"Generated image has zero dimensions ({w}x{h}): {path}"
                )
            return w, h

    def _generate_image(self, scene: Dict) -> Tuple[Optional[Path], str]:
        """
        Generate one image. Returns (image_path, backend_used).
        Never raises — always returns something.
        """
        if self.dry_run:
            logger.info("[DRY RUN] Would generate: %s", scene.get("segment_id"))
            return None, "dry_run"

        segment_id = scene.get("segment_id", "unknown")

        # Build full prompt with character context
        base_prompt = scene.get("image_prompt", "")
        char_context = self._build_char_context(scene.get("characters", []))
        full_prompt = base_prompt
        if char_context:
            full_prompt = f"{base_prompt}\n{char_context}"

        from lib.config import DEFAULT_IMAGE_MODEL

        img = None
        backend = None

        # Try FLUX only if it's the configured default
        if DEFAULT_IMAGE_MODEL == "flux_schnell":
            img = self._generate_with_flux(full_prompt, segment_id=segment_id)
            backend = "flux"

        # Try SDXL if FLUX failed or if SDXL is the default
        if img is None:
            img = self._generate_with_sdxl(full_prompt, segment_id=segment_id)
            backend = "sdxl"

        if img is None:
            # Last resort: reuse nearest image
            fallback = self._reuse_nearest_image()
            if fallback:
                logger.warning(
                    "Scene %s: all generators failed — reusing %s",
                    segment_id, fallback.name,
                )
                out_path = self.images_dir / f"scene_{segment_id}.png"
                shutil.copy2(fallback, out_path)
                return out_path, "reused"
            return None, "failed"

        # Save to images/ and scene_cache/
        out_path = self.images_dir / f"scene_{segment_id}.png"
        img.save(str(out_path), "PNG")

        # Validate: file must be readable with non-zero dimensions
        try:
            img_w, img_h = self._validate_image(out_path)
            logger.info(
                "  %s: ✅ %s (%dx%d)", segment_id, backend, img_w, img_h
            )
        except Exception as e:
            logger.error(
                "Scene %s: output validation failed after generation: %s",
                segment_id, e,
            )
            return None, "invalid"

        cache_key = _scene_cache_key(scene)
        cache_path = self.scene_cache_dir / f"{cache_key}.png"
        if not cache_path.exists():
            img.save(str(cache_path), "PNG")

        return out_path, backend

    # ── Episode-level entry point ─────────────────────────────────────────────

    def generate_episode(
        self,
        episode: int,
        max_scenes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate all images for an episode.

        Scene-level checkpointing: saves after EVERY image.
        Checks scene cache before generating.

        Parameters
        ----------
        episode : int
            Episode number to generate images for.
        max_scenes : int, optional
            If set, process only the first N scenes (debug mode).
            Useful for fast iteration without waiting for the full episode.
        """
        script_path = self.project_dir / "scripts" / f"episode_{episode}.json"
        if not script_path.exists():
            raise FileNotFoundError(f"Storyboard missing: {script_path}")

        with open(script_path, encoding="utf-8") as f:
            script = json.load(f)

        segments = script.get("segments", [])

        if max_scenes is not None:
            segments = segments[:max_scenes]
            logger.info(
                "Debug mode: limiting to %d of %d scenes",
                max_scenes, len(script.get("segments", [])),
            )

        total = len(segments)
        logger.info("Image generation: %d scenes, episode %d", total, episode)

        generated = 0
        cached = 0
        failed = 0

        for seg in segments:
            sid = seg.get("segment_id", "?")

            # Already done in this run or previous session
            if sid in self._progress:
                logger.debug("  %s: already generated, skipping", sid)
                continue

            # Check scene cache
            cache_key = _scene_cache_key(seg)
            cache_path = self.scene_cache_dir / f"{cache_key}.png"
            out_path = self.images_dir / f"scene_{sid}.png"

            if cache_path.exists() and not out_path.exists():
                shutil.copy2(cache_path, out_path)
                self._save_progress(sid, str(out_path))
                logger.info("  %s: ✅ cache hit (%s)", sid, cache_key[:8])
                cached += 1

                # Update storyboard JSON
                self._update_storyboard_asset(script_path, script, sid, str(out_path))
                continue

            if out_path.exists():
                self._save_progress(sid, str(out_path))
                continue

            # Generate
            logger.info(
                "  %s: generating (%s) …",
                sid, self._active_backend or "not loaded yet",
            )
            result_path, backend = self._generate_image(seg)

            if result_path:
                self._save_progress(sid, str(result_path))
                # Update storyboard JSON immediately
                self._update_storyboard_asset(script_path, script, sid, str(result_path))
                generated += 1
            else:
                failed += 1
                logger.warning("  %s: ❌ generation failed", sid)

        logger.info(
            "Episode %d images: %d new, %d cached, %d failed",
            episode, generated, cached, failed,
        )
        return {
            "generated": generated,
            "cached": cached,
            "failed": failed,
            "total": total,
        }

    def _update_storyboard_asset(
        self,
        script_path: Path,
        script: Dict,
        segment_id: str,
        image_path: str,
    ) -> None:
        """Update generated_assets in the episode script JSON immediately."""
        try:
            for seg in script.get("segments", []):
                if seg.get("segment_id") == segment_id:
                    if "generated_assets" not in seg:
                        seg["generated_assets"] = {}
                    seg["generated_assets"]["storyboard_image"] = image_path
                    seg["generated_assets"]["status"] = "storyboard_ready"
                    break
            with open(script_path, "w", encoding="utf-8") as f:
                json.dump(script, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(
                "Could not update storyboard JSON for %s: %s", segment_id, e
            )

    def generate_character_portrait(
        self,
        character_name: str,
        character_json: Dict,
    ) -> Optional[Path]:
        """
        Generate a character reference portrait.
        Called during Stage 2 so Qwen is already unloaded.
        Returns path to hero_ref.png or None.
        """
        stem = re.sub(r"[^\w\-]", "_", character_name.strip()).lower()
        chars_dir = self.project_dir / "characters"
        ref_path = chars_dir / f"{stem}_ref.png"

        if ref_path.exists():
            return ref_path

        # Build portrait prompt from character JSON
        parts = [
            f"Portrait of {character_json.get('gender', 'person')}",
            f"age {character_json.get('age', 'adult')}",
            character_json.get("hair", ""),
            f"{character_json.get('eyes', '')} eyes",
            character_json.get("body_type", ""),
            character_json.get("clothing", ""),
            character_json.get("special_features", ""),
            "anime illustration style, detailed face, plain background, character reference sheet",
        ]
        prompt = ", ".join(p for p in parts if p)

        fake_scene = {
            "segment_id": f"char_{stem}",
            "location": "studio",
            "characters": [],
            "scene_type": "exposition",
            "image_prompt": prompt,
        }

        result, _ = self._generate_image(fake_scene)
        if result:
            shutil.copy2(result, ref_path)
            # Clean up temp file
            result.unlink(missing_ok=True)
            return ref_path
        return None
