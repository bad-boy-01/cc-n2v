"""
lib/media_generator.py — DEPRECATED (v2)

This module has been removed as part of the CC-Novel2Video v2 refactoring.

Replacements
------------
- Image generation  → lib/services/image_service.py
- Video generation  → lib/services/video_service.py
- Character images  → lib/services/image_service.generate_character_portrait()
- Clue images       → lib/services/image_service.generate_clue_image()

All model access goes through:
- lib/image_generator.ImageGenerator  (FLUX.1-schnell / SDXL)
- lib/motion_engine.MotionEngine       (Ken Burns / pan effects)
"""

raise ImportError(
    "\n\nMediaGenerator has been removed in CC-Novel2Video v2.\n"
    "Use the following replacements:\n"
    "  - Image generation:   lib/services/image_service.py\n"
    "  - Video generation:   lib/services/video_service.py\n"
    "  - Character portrait: image_service.generate_character_portrait()\n"
    "  - Clue image:         image_service.generate_clue_image()\n"
)
