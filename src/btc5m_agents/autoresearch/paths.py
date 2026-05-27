"""Paths for prompt autoresearch artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from btc5m_agents.config import Settings, get_settings


def autoresearch_dir(settings: Optional[Settings] = None) -> Path:
    s = settings or get_settings()
    return s.project_root / "autoresearch"


def prompts_path(settings: Optional[Settings] = None) -> Path:
    s = settings or get_settings()
    return s.project_root / "src" / "btc5m_agents" / "agents" / "prompts.py"


def prompts_best_path(settings: Optional[Settings] = None) -> Path:
    return autoresearch_dir(settings) / "prompts_best.py"


def last_eval_path(settings: Optional[Settings] = None) -> Path:
    return autoresearch_dir(settings) / "last_eval.json"


def experiments_path(settings: Optional[Settings] = None) -> Path:
    return autoresearch_dir(settings) / "experiments.jsonl"


def best_mean_score_path(settings: Optional[Settings] = None) -> Path:
    return autoresearch_dir(settings) / "best_mean_score.txt"


def prompts_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
