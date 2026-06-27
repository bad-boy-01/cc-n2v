# CC-Novel2Video Frontend Logic Audit Report

## Summary

| Severity | Count |
|---|---|
| 🔴 High (business logic in UI layer) | 10 |
| 🟡 Medium (logic that should move to services) | 0 |
| 🟢 Low (minor references) | 222 |
| **Total** | **232** |

---

## Migration Priority

### 🔴 High Priority — Move to `lib/services/` immediately

#### `webui\server\routers\generate.py`
- **Line 78** — generate_video() call
  ```
  async def generate_video(
  ```

#### `webui\server\services\generation_tasks.py`
- **Line 25** — execute_storyboard_task() — task logic
  ```
  def execute_storyboard_task(
  ```
- **Line 52** — execute_video_task() — task logic
  ```
  def execute_video_task(
  ```
- **Line 97** — execute_character_task() — task logic
  ```
  def execute_character_task(
  ```
- **Line 113** — execute_clue_task() — task logic
  ```
  def execute_clue_task(
  ```
- **Line 154** — execute_storyboard_task() — task logic
  ```
  result = execute_storyboard_task(
  ```
- **Line 186** — execute_storyboard_task() — task logic
  ```
  return execute_storyboard_task(project_name, str(resource_id), payload)
  ```
- **Line 188** — execute_video_task() — task logic
  ```
  return execute_video_task(project_name, str(resource_id), payload)
  ```
- **Line 190** — execute_character_task() — task logic
  ```
  return execute_character_task(project_name, str(resource_id), payload)
  ```
- **Line 192** — execute_clue_task() — task logic
  ```
  return execute_clue_task(project_name, str(resource_id), payload)
  ```

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
