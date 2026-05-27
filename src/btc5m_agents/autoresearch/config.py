"""Load autoresearch config.yaml (simple key: value, no PyYAML required)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from btc5m_agents.config import Settings, get_settings


def load_autoresearch_config(path: Path, settings: Optional[Settings] = None) -> dict[str, Any]:
    s = settings or get_settings()
    p = path if path.is_absolute() else s.project_root / path
    if not p.exists():
        raise FileNotFoundError(f"Autoresearch config not found: {p}")

    out: dict[str, Any] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, val = stripped.split(":", 1)
        key = key.strip()
        val = val.strip()
        if val.lower() in ("true", "false"):
            out[key] = val.lower() == "true"
        else:
            try:
                if "." in val:
                    out[key] = float(val)
                else:
                    out[key] = int(val)
            except ValueError:
                out[key] = val
    return out


def resolve_path(value: str, settings: Optional[Settings] = None) -> Path:
    s = settings or get_settings()
    p = Path(value)
    if p.is_absolute():
        return p
    return s.project_root / p
