"""
lib/world_engine.py — Persistent fictional world state tracker.

Tracks the full world across all scenes and episodes, giving every downstream
module access to coherent, up-to-date character and location information.

This eliminates continuity issues (outfit changes, injuries, location state)
that would otherwise require prompt duplication or post-hoc fixing.

Storage: projects/{name}/project_world.json

World State Structure
---------------------
characters:   Visual + narrative state per character
locations:    Atmosphere, lighting, visual description
timeline:     Ordered list of plot events
kingdoms:     Political entities
organizations: Factions, guilds, armies
magic_system: Rules, limitations, power tiers
active_events: Currently unresolved plot events
current_*:    Global scene state (location, time, weather)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Default character template ────────────────────────────────────────────────

_DEFAULT_CHARACTER: Dict[str, Any] = {
    "appearance": "",
    "clothing": "",
    "hair": "",
    "eyes": "",
    "age": "",
    "alive": True,
    "power_level": "",
    "current_goal": "",
    "injuries": [],
    "weapons": [],
    "inventory": [],
    "emotional_state": "neutral",
    "relationships": {},
    "last_seen": "",
    "last_scene": "",
    "last_outfit_change": "",
    "first_appearance_scene": "",
}

# ── Default world state ───────────────────────────────────────────────────────

_DEFAULT_WORLD: Dict[str, Any] = {
    "characters": {},
    "locations": {},
    "timeline": [],
    "kingdoms": {},
    "organizations": {},
    "magic_system": {},
    "active_events": [],
    "current_chapter": 1,
    "current_location": "",
    "current_time_of_day": "day",
    "current_weather": "clear",
}


class WorldEngine:
    """
    Persistent world-state tracker for long-form video generation.

    All agents interact with the world through this class — they do not
    read/write project_world.json directly.

    Usage
    -----
    engine = WorldEngine("my_project", "projects")
    engine.load()

    # During character extraction:
    engine.register_character("Hero", {
        "hair": "black", "eyes": "blue", "clothing": "dark robe"
    })

    # Before building image prompt:
    context = engine.get_full_context(scene)

    # After scene processed:
    engine.update_from_scene(scene)
    """

    def __init__(self, project_name: str, projects_root: Optional[str] = None):
        """
        Parameters
        ----------
        project_name : str
        projects_root : str, optional
            Root directory for projects (default: "projects")
        """
        root = Path(projects_root) if projects_root else Path("projects")
        self.project_dir = root / project_name
        self._path = self.project_dir / "project_world.json"
        self._world: Dict[str, Any] = {}
        self._dirty = False

    # ── Persistence ───────────────────────────────────────────────────────────

    def load(self) -> "WorldEngine":
        """Load world state from disk. Returns self for chaining."""
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    self._world = json.load(f)
                logger.info(f"[WorldEngine] Loaded: {self._path}")
            except Exception as e:
                logger.warning(f"[WorldEngine] Failed to load, using defaults: {e}")
                self._world = json.loads(json.dumps(_DEFAULT_WORLD))
        else:
            self._world = json.loads(json.dumps(_DEFAULT_WORLD))
        return self

    def save(self) -> None:
        """Persist world state to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._world, f, ensure_ascii=False, indent=2)
        self._dirty = False
        logger.debug(f"[WorldEngine] Saved: {self._path}")

    def save_if_dirty(self) -> None:
        """Save only if state has changed since last save."""
        if self._dirty:
            self.save()

    # ── Character management ──────────────────────────────────────────────────

    def register_character(
        self,
        name: str,
        data: Dict[str, Any],
        overwrite: bool = False,
    ) -> None:
        """
        Register or update a character's world state.

        Called by CharacterAgent after extraction.
        Only updates fields that are present in `data` — existing fields
        are preserved unless overwrite=True.
        """
        chars = self._world.setdefault("characters", {})
        if name not in chars or overwrite:
            chars[name] = dict(_DEFAULT_CHARACTER)

        char = chars[name]
        # Map common CharacterAgent fields
        field_map = {
            "hair": "hair", "eyes": "eyes", "age": "age",
            "clothing": "clothing", "body_type": "appearance",
            "special_features": "appearance",
            "gender": "appearance",
        }
        for src, dst in field_map.items():
            if data.get(src):
                if dst == "appearance" and char.get("appearance"):
                    char[dst] += f", {data[src]}"
                else:
                    char[dst] = data[src]

        # Direct field copies
        for field in ["weapons", "injuries", "inventory", "emotional_state",
                      "power_level", "current_goal", "relationships"]:
            if field in data:
                char[field] = data[field]

        self._dirty = True
        logger.debug(f"[WorldEngine] Registered character: {name}")

    def get_character(self, name: str) -> Dict[str, Any]:
        """Return character state dict. Returns empty dict if not found."""
        return self._world.get("characters", {}).get(name, {})

    def mark_character_change(
        self,
        name: str,
        field: str,
        value: Any,
        scene_id: str,
    ) -> None:
        """
        Update a single character field and log the change to timeline.

        Used to track outfit changes, injuries, weapon acquisitions, etc.
        """
        chars = self._world.setdefault("characters", {})
        if name not in chars:
            chars[name] = dict(_DEFAULT_CHARACTER)

        old_value = chars[name].get(field)
        chars[name][field] = value
        chars[name]["last_scene"] = scene_id

        if field == "clothing":
            chars[name]["last_outfit_change"] = scene_id

        # Log to timeline
        self._world.setdefault("timeline", []).append({
            "scene": scene_id,
            "event": f"{name}: {field} changed from '{old_value}' to '{value}'",
            "type": "character_change",
        })
        self._dirty = True

    # ── Location management ───────────────────────────────────────────────────

    def register_location(
        self,
        name: str,
        data: Dict[str, Any],
    ) -> None:
        """Register or update a location. Called by ClueAgent."""
        locs = self._world.setdefault("locations", {})
        locs[name] = {
            "description": data.get("visual_description", data.get("description", "")),
            "atmosphere": data.get("atmosphere", ""),
            "type": data.get("type", "location"),
            "time_of_day": data.get("time_of_day", ""),
            "weather": data.get("weather", ""),
        }
        self._dirty = True
        logger.debug(f"[WorldEngine] Registered location: {name}")

    def get_location(self, name: str) -> Dict[str, Any]:
        """Return location state dict."""
        return self._world.get("locations", {}).get(name, {})

    # ── Scene state management ────────────────────────────────────────────────

    def update_from_scene(self, scene: Dict[str, Any]) -> None:
        """
        Update world state after a scene is processed.

        Called by StoryboardAgent after each scene is finalized.
        Updates: current location, time of day, character last_seen, timeline.
        """
        scene_id = scene.get("segment_id", "")
        location = scene.get("location", "")
        time_of_day = scene.get("time_of_day", "")
        weather = scene.get("weather", "")
        characters = scene.get("characters", [])

        if location:
            self._world["current_location"] = location
        if time_of_day:
            self._world["current_time_of_day"] = time_of_day
        if weather:
            self._world["current_weather"] = weather

        # Update each character's last_seen
        for char_name in characters:
            if char_name in self._world.get("characters", {}):
                self._world["characters"][char_name]["last_seen"] = location
                self._world["characters"][char_name]["last_scene"] = scene_id

        # Log to timeline
        if scene_id:
            summary = scene.get("title", scene_id)
            self._world.setdefault("timeline", []).append({
                "scene": scene_id,
                "event": summary,
                "location": location,
                "characters": characters,
                "chapter": self._world.get("current_chapter", 1),
                "type": "scene",
            })

        self._dirty = True

    # ── Prompt context builders ───────────────────────────────────────────────

    def get_character_context(self, character_names: List[str]) -> str:
        """
        Build character visual context string for prompt injection.

        Returns a compact multi-line description of each character's
        current visual state (clothing, injuries, weapons, etc.).
        """
        parts = []
        for name in character_names:
            char = self.get_character(name)
            if not char:
                continue
            desc_parts = []
            if char.get("hair"):
                desc_parts.append(f"{char['hair']} hair")
            if char.get("eyes"):
                desc_parts.append(f"{char['eyes']} eyes")
            if char.get("clothing"):
                desc_parts.append(f"wearing {char['clothing']}")
            if char.get("injuries"):
                desc_parts.append(f"injured: {', '.join(char['injuries'])}")
            if char.get("weapons"):
                desc_parts.append(f"carrying {', '.join(char['weapons'])}")
            if desc_parts:
                parts.append(f"{name}: {', '.join(desc_parts)}")
        return "\n".join(parts)

    def get_location_context(self, location_name: str) -> str:
        """Build location atmosphere string for prompt injection."""
        loc = self.get_location(location_name)
        if not loc:
            return ""
        parts = []
        if loc.get("description"):
            parts.append(loc["description"])
        if loc.get("atmosphere"):
            parts.append(loc["atmosphere"])
        return ". ".join(parts)

    def get_full_context(self, scene: Dict[str, Any]) -> str:
        """
        Build the complete world context string for a scene.

        Includes: characters, location, time of day, weather.
        Intended for injection into storyboard prompts.
        """
        parts = []

        char_ctx = self.get_character_context(scene.get("characters", []))
        if char_ctx:
            parts.append(char_ctx)

        loc_ctx = self.get_location_context(scene.get("location", ""))
        if loc_ctx:
            parts.append(f"Location — {loc_ctx}")

        time_of_day = scene.get("time_of_day", self._world.get("current_time_of_day", ""))
        if time_of_day:
            parts.append(f"Time: {time_of_day}")

        weather = scene.get("weather", self._world.get("current_weather", ""))
        if weather and weather != "clear":
            parts.append(f"Weather: {weather}")

        return "\n".join(parts)

    # ── World-level accessors ─────────────────────────────────────────────────

    def add_timeline_event(
        self,
        scene_id: str,
        event: str,
        event_type: str = "event",
    ) -> None:
        """Manually log a significant plot event."""
        self._world.setdefault("timeline", []).append({
            "scene": scene_id,
            "event": event,
            "type": event_type,
            "chapter": self._world.get("current_chapter", 1),
        })
        self._dirty = True

    def set_chapter(self, chapter: int) -> None:
        self._world["current_chapter"] = chapter
        self._dirty = True

    def add_active_event(self, event: str) -> None:
        self._world.setdefault("active_events", []).append(event)
        self._dirty = True

    def resolve_event(self, event: str) -> None:
        events = self._world.get("active_events", [])
        if event in events:
            events.remove(event)
        self._dirty = True

    def snapshot(self) -> Dict[str, Any]:
        """Return full world state (for ManifestBuilder)."""
        return dict(self._world)

    @property
    def character_count(self) -> int:
        return len(self._world.get("characters", {}))

    @property
    def location_count(self) -> int:
        return len(self._world.get("locations", {}))

    @property
    def current_location(self) -> str:
        return self._world.get("current_location", "")

    @property
    def current_time_of_day(self) -> str:
        return self._world.get("current_time_of_day", "day")
