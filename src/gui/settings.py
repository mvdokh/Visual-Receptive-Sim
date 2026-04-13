"""Persistent user preferences (JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_KNOWN_KEYS = frozenset({"active_theme_id", "active_preset", "last_export_dir"})

_SETTINGS_DIR = Path.home() / ".visual_receptive_sim"
_SETTINGS_PATH = _SETTINGS_DIR / "user_settings.json"

_DEFAULTS: dict[str, Any] = {
    "active_theme_id": "dark_plus",
    "active_preset": "default",
    "last_export_dir": str(Path.home()),
}


def settings_path() -> Path:
    return _SETTINGS_PATH


def load() -> dict[str, Any]:
    """Return merged settings; unknown keys in the file are ignored."""
    out = dict(_DEFAULTS)
    try:
        if _SETTINGS_PATH.is_file():
            raw = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if k in _KNOWN_KEYS:
                        out[k] = v
    except (OSError, json.JSONDecodeError):
        pass
    return out


def save(data: dict[str, Any]) -> None:
    """Persist known keys only."""
    _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {k: data[k] for k in _KNOWN_KEYS if k in data}
    _SETTINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
