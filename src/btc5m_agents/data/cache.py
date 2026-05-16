"""Deterministic Parquet read/write helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from btc5m_agents.config import Settings, get_settings


def cache_root(settings: Optional[Settings] = None) -> Path:
    s = settings or get_settings()
    root = s.resolved_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
