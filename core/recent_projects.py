"""NexStitch — Recent projects list.

Stored at ~/.nexstitch/recent.json  (outside any project folder)
so it persists across sessions regardless of which project is open.
"""
from __future__ import annotations

import json
from pathlib import Path

_CFG_DIR     = Path.home() / ".nexstitch"
_RECENT_FILE = _CFG_DIR / "recent.json"
_MAX         = 10


def load_recent() -> list[dict]:
    """Return list of {nxs, name, modified} dicts, newest first."""
    if not _RECENT_FILE.exists():
        return []
    try:
        data = json.loads(_RECENT_FILE.read_text("utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def add_recent(project) -> None:
    """Insert / move project to the top of the recent list."""
    _CFG_DIR.mkdir(parents=True, exist_ok=True)
    entries  = load_recent()
    nxs_str  = str(project.nxs_path)
    entries  = [e for e in entries if e.get("nxs") != nxs_str]
    entries.insert(0, {
        "nxs":      nxs_str,
        "name":     project.name,
        "modified": project.modified,
    })
    _RECENT_FILE.write_text(
        json.dumps(entries[:_MAX], indent=2, ensure_ascii=False), "utf-8"
    )


def remove_recent(nxs_str: str) -> None:
    """Remove one entry from the recent list."""
    entries = [e for e in load_recent() if e.get("nxs") != nxs_str]
    _CFG_DIR.mkdir(parents=True, exist_ok=True)
    _RECENT_FILE.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), "utf-8"
    )
