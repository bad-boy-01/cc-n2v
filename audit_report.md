# CC-Novel2Video Frontend Logic Audit Report

## Summary

| Severity | Count |
|---|---|
| 🔴 High (business logic in UI layer) | 0 |
| 🟡 Medium (logic that should move to services) | 0 |
| 🟢 Low (minor references) | 259 |
| **Total** | **259** |

---

## Migration Priority

### 🔴 High Priority — Move to `lib/services/` immediately

_No high-severity findings._

### 🟡 Medium Priority — Extract to service functions

_No medium-severity findings._

---

## Recommended Service Mapping

| Current Location | Move To |
|---|---|
| `webui/server/services/generation_tasks.py` → `execute_storyboard_task()` | `lib/services/storyboard_service.py` |
| `webui/server/services/generation_tasks.py` → `execute_video_task()` | `lib/services/video_service.py` |
| `webui/server/services/generation_tasks.py` → `execute_character_task()` | `lib/services/image_service.py` |
| `webui/server/services/generation_tasks.py` → `execute_clue_task()` | `lib/services/image_service.py` |
| `webui/server/services/generation_tasks.py` → `_build_grid_prompt()` | `lib/services/storyboard_service.py` |
| `webui/server/services/generation_tasks.py` → `_collect_reference_images()` | `lib/services/storyboard_service.py` |
| `webui/server/services/generation_tasks.py` → `_normalize_storyboard_prompt()` | `lib/services/storyboard_service.py` |
| `lib/gemini_client.py` (all image generation) | `lib/services/image_service.py` → `lib/image_generator.py` |
| Veo video generation | DELETE — replaced by `lib/motion_engine.py` |

---

## Target Architecture

```
WebUI (thin layer)              CLI / Notebook
     |                               |
     └─────────────┬─────────────────┘
                   |
            lib/services/
         ┌─────────┴──────────┐
         |                    |
  storyboard_service   pipeline_service
  image_service        audio_service
  video_service        subtitle_service
         |
   lib/agents/ + lib/image_generator.py
   lib/motion_engine.py + lib/kokoro_tts.py
```

After migration: No model calls, FFmpeg calls, or OCR calls inside `webui/`.
