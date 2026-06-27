"""
lib/review_generator.py — Static HTML review page generator.

Generates output/review.html — a self-contained, offline-compatible
page showing every generated scene with:
  - Scene image
  - Narration text
  - Full prompt used
  - Duration, camera motion, difficulty score
  - Quality check result
  - "Regenerate" deep-link to WebUI

No server required. Works in Kaggle notebooks and local browsers.
"""

from __future__ import annotations

import html
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: #0f0f13;
  color: #e0e0e8;
  padding: 24px;
}
h1 { color: #a78bfa; font-size: 28px; margin-bottom: 8px; }
.subtitle { color: #888; font-size: 14px; margin-bottom: 32px; }
.stats-bar {
  display: flex; gap: 24px; margin-bottom: 32px;
  background: #1a1a24; border-radius: 12px; padding: 16px 24px;
}
.stat { display: flex; flex-direction: column; }
.stat-val { font-size: 24px; font-weight: 700; color: #a78bfa; }
.stat-lbl { font-size: 12px; color: #888; margin-top: 2px; }
.filter-bar {
  display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap;
}
.filter-btn {
  padding: 6px 16px; border-radius: 20px; border: 1px solid #333;
  background: #1a1a24; color: #aaa; cursor: pointer; font-size: 13px;
  transition: all .2s;
}
.filter-btn.active { background: #a78bfa; color: #fff; border-color: #a78bfa; }
.search-box {
  padding: 6px 14px; border-radius: 20px; border: 1px solid #333;
  background: #1a1a24; color: #e0e0e8; font-size: 13px; width: 240px;
}
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 20px; }
.card {
  background: #1a1a24; border-radius: 16px; overflow: hidden;
  border: 1px solid #2a2a38; transition: transform .2s, border-color .2s;
}
.card:hover { transform: translateY(-3px); border-color: #a78bfa44; }
.card-img { width: 100%; aspect-ratio: 16/9; object-fit: cover;
  background: #111; display: block; }
.card-img-placeholder {
  width: 100%; aspect-ratio: 16/9; background: #111;
  display: flex; align-items: center; justify-content: center;
  color: #444; font-size: 13px;
}
.card-body { padding: 14px 16px; }
.card-header {
  display: flex; justify-content: space-between; align-items: flex-start;
  margin-bottom: 8px;
}
.scene-id { font-size: 11px; color: #666; font-family: monospace; }
.badges { display: flex; gap: 6px; flex-wrap: wrap; }
.badge {
  font-size: 10px; padding: 2px 8px; border-radius: 10px;
  font-weight: 600; text-transform: uppercase; letter-spacing: .5px;
}
.badge-pass { background: #14532d; color: #4ade80; }
.badge-fail { background: #7f1d1d; color: #f87171; }
.badge-warn { background: #713f12; color: #fbbf24; }
.badge-easy { background: #1e3a5f; color: #60a5fa; }
.badge-medium { background: #312e81; color: #818cf8; }
.badge-hard { background: #4c1d95; color: #c4b5fd; }
.badge-extreme { background: #7f1d1d; color: #f87171; }
.badge-flux { background: #1a2744; color: #93c5fd; }
.badge-sdxl { background: #1a3a2a; color: #86efac; }
.card-text {
  font-size: 13px; color: #bbb; margin-bottom: 10px;
  line-height: 1.6; max-height: 80px; overflow: hidden;
  position: relative;
}
.card-text::after {
  content: ''; position: absolute; bottom: 0; left: 0; right: 0;
  height: 24px;
  background: linear-gradient(transparent, #1a1a24);
}
.prompt-box {
  background: #111; border-radius: 8px; padding: 8px 10px;
  font-size: 11px; color: #666; font-family: monospace;
  max-height: 60px; overflow: hidden; margin-bottom: 10px;
  position: relative;
}
.prompt-box::after {
  content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 20px;
  background: linear-gradient(transparent, #111);
}
.card-meta {
  display: flex; gap: 12px; font-size: 11px; color: #555;
  margin-bottom: 10px; flex-wrap: wrap;
}
.meta-item { display: flex; align-items: center; gap: 4px; }
.regen-btn {
  width: 100%; padding: 8px; border-radius: 8px;
  background: #2a1f5f; color: #a78bfa; border: 1px solid #3d2f8f;
  cursor: pointer; font-size: 12px; font-weight: 600; text-align: center;
  text-decoration: none; display: block; transition: background .2s;
}
.regen-btn:hover { background: #3d2f8f; }
.issue-list { font-size: 11px; color: #f87171; margin-bottom: 8px; }
"""

_JS = """
function filterCards(quality) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  const search = document.getElementById('searchBox').value.toLowerCase();
  document.querySelectorAll('.card').forEach(c => {
    const matchQ = quality === 'all' || c.dataset.quality === quality;
    const matchS = !search || c.dataset.search.includes(search);
    c.style.display = (matchQ && matchS) ? '' : 'none';
  });
}
function searchCards() {
  const search = document.getElementById('searchBox').value.toLowerCase();
  const activeQ = document.querySelector('.filter-btn.active')?.dataset.quality || 'all';
  document.querySelectorAll('.card').forEach(c => {
    const matchQ = activeQ === 'all' || c.dataset.quality === activeQ;
    const matchS = !search || c.dataset.search.includes(search);
    c.style.display = (matchQ && matchS) ? '' : 'none';
  });
}
"""


class ReviewGenerator:
    """
    Static HTML review page generator.

    Usage
    -----
    gen = ReviewGenerator("my_project", "projects")
    path = gen.generate(episode=1, scenes=scenes, quality_report=report)
    """

    def __init__(self, project_name: str, projects_root: Optional[str] = None):
        root = Path(projects_root) if projects_root else Path("projects")
        self.project_dir = root / project_name
        self.project_name = project_name

    def generate(
        self,
        episode: int,
        scenes: Optional[List[Dict]] = None,
        quality_report: Optional[Any] = None,
        asset_db: Optional[Any] = None,
        webui_base_url: str = "http://localhost:5000",
    ) -> Optional[Path]:
        """
        Generate review.html.

        Parameters
        ----------
        episode : int
        scenes : list, optional
            List of scene dicts from episode JSON
        quality_report : EpisodeReport, optional
        asset_db : AssetDatabase, optional
        webui_base_url : str
            Base URL for regenerate links (default localhost)

        Returns
        -------
        Path | None
        """
        # Load scenes from disk if not provided
        if not scenes:
            scenes = self._load_scenes(episode)

        if not scenes:
            logger.warning("[ReviewGenerator] No scenes found — skipping review page")
            return None

        # Build quality index
        quality_index: Dict[str, str] = {}
        quality_issues: Dict[str, List[str]] = {}
        if quality_report:
            for r in quality_report.images:
                scene_id = Path(r.path).stem
                quality_index[scene_id] = r.severity
                quality_issues[scene_id] = r.issues

        # Build asset db index
        backend_index: Dict[str, str] = {}
        if asset_db:
            for img in asset_db.get_generated_images():
                backend_index[img["scene_id"]] = img.get("backend", "")

        # Statistics
        n_pass = sum(1 for s in scenes if quality_index.get(s.get("segment_id", ""), "pass") == "pass")
        n_fail = sum(1 for s in scenes if quality_index.get(s.get("segment_id", ""), "pass") == "fail")

        # Build HTML
        html_content = self._build_html(
            episode, scenes, quality_index, quality_issues,
            backend_index, n_pass, n_fail, webui_base_url
        )

        # Save
        output_dir = self.project_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"review_episode_{episode}.html"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"[ReviewGenerator] Review page saved: {out_path}")
        return out_path

    def _load_scenes(self, episode: int) -> List[Dict]:
        """Load scenes from episode JSON script."""
        script_path = self.project_dir / "scripts" / f"episode_{episode}.json"
        if not script_path.exists():
            return []
        try:
            with open(script_path, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("segments", [])
        except Exception:
            return []

    def _build_html(
        self,
        episode: int,
        scenes: List[Dict],
        quality_index: Dict,
        quality_issues: Dict,
        backend_index: Dict,
        n_pass: int,
        n_fail: int,
        webui_url: str,
    ) -> str:
        """Build the full HTML document."""
        title = f"{self.project_name} — Episode {episode} Review"
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        cards_html = "\n".join(
            self._build_card(s, quality_index, quality_issues, backend_index, webui_url)
            for s in scenes
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(title)}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>🎬 {html.escape(title)}</h1>
  <div class="subtitle">Generated: {now} · {len(scenes)} scenes</div>

  <div class="stats-bar">
    <div class="stat"><span class="stat-val">{len(scenes)}</span><span class="stat-lbl">Total Scenes</span></div>
    <div class="stat"><span class="stat-val" style="color:#4ade80">{n_pass}</span><span class="stat-lbl">Passed QC</span></div>
    <div class="stat"><span class="stat-val" style="color:#f87171">{n_fail}</span><span class="stat-lbl">Failed QC</span></div>
    <div class="stat"><span class="stat-val">{round(sum(s.get('duration_seconds', 4) for s in scenes) / 60, 1)} min</span><span class="stat-lbl">Est. Runtime</span></div>
  </div>

  <div class="filter-bar">
    <input class="search-box" id="searchBox" placeholder="Search scenes…" oninput="searchCards()">
    <button class="filter-btn active" data-quality="all" onclick="filterCards('all')">All</button>
    <button class="filter-btn" data-quality="pass" onclick="filterCards('pass')">✅ Passed</button>
    <button class="filter-btn" data-quality="fail" onclick="filterCards('fail')">❌ Failed</button>
    <button class="filter-btn" data-quality="warn" onclick="filterCards('warn')">⚠️ Warning</button>
  </div>

  <div class="grid" id="sceneGrid">{cards_html}</div>

  <script>{_JS}</script>
</body>
</html>"""

    def _build_card(
        self,
        scene: Dict,
        quality_index: Dict,
        quality_issues: Dict,
        backend_index: Dict,
        webui_url: str,
    ) -> str:
        """Build a single scene card."""
        scene_id = scene.get("segment_id", "")
        quality = quality_index.get(scene_id, "pass")
        issues = quality_issues.get(scene_id, [])
        backend = backend_index.get(scene_id, "flux")
        difficulty = scene.get("difficulty", "medium")
        camera = scene.get("camera_motion", "Static")
        duration = scene.get("duration_seconds", 4.0)
        narration = html.escape(scene.get("novel_text", "")[:300])
        prompt = html.escape(scene.get("image_prompt", "")[:250])
        location = html.escape(scene.get("location", ""))
        chars = ", ".join(html.escape(c) for c in scene.get("characters", []))

        # Image path
        images_dir = self.project_dir / "images"
        img_path = None
        for ext in [".png", ".jpg", ".webp"]:
            candidate = images_dir / f"{scene_id}{ext}"
            if candidate.exists():
                # Use relative path for portability
                img_path = f"../images/{scene_id}{ext}"
                break

        img_html = (
            f'<img class="card-img" src="{img_path}" alt="{scene_id}" loading="lazy">'
            if img_path else
            f'<div class="card-img-placeholder">No image</div>'
        )

        quality_badge = f'<span class="badge badge-{quality}">{quality.upper()}</span>'
        diff_badge = f'<span class="badge badge-{difficulty}">{difficulty}</span>'
        backend_badge = f'<span class="badge badge-{backend}">{backend}</span>'

        issues_html = ""
        if issues:
            issues_html = '<div class="issue-list">⚠️ ' + " | ".join(html.escape(i) for i in issues) + "</div>"

        regen_url = f"{webui_url}/projects/{self.project_name}/scenes/{scene_id}/regenerate"

        search_data = f"{scene_id} {location} {chars} {quality}".lower()

        return f"""<div class="card" data-quality="{quality}" data-search="{html.escape(search_data)}">
  {img_html}
  <div class="card-body">
    <div class="card-header">
      <span class="scene-id">{scene_id}</span>
      <div class="badges">{quality_badge}{diff_badge}{backend_badge}</div>
    </div>
    <div class="card-meta">
      <span class="meta-item">📍 {location}</span>
      <span class="meta-item">⏱ {duration:.1f}s</span>
      <span class="meta-item">🎥 {camera}</span>
      {f'<span class="meta-item">👥 {chars}</span>' if chars else ''}
    </div>
    <div class="card-text">{narration}</div>
    <div class="prompt-box">{prompt}</div>
    {issues_html}
    <a class="regen-btn" href="{regen_url}" target="_blank">🔄 Regenerate Scene</a>
  </div>
</div>"""
