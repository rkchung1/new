"""Load and normalize btc_5m_2s replay data for the backtest engine."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from btc5m_agents.config import Settings, get_settings
from btc5m_agents.data.cache import cache_root, read_parquet, write_parquet
from btc5m_agents.data.market_discovery import WINDOW_DURATION_SEC

_RAW_COLS = {
    "slug",
    "start_time",
    "elapsed",
    "ask_YES",
    "bid_YES",
    "timestamp_log",
    "btc_strike",
    "btc_current",
    "btc_gap",
    "winner",
}


def _parse_day(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def _utc_range_ts(start_date: str, end_date: str) -> tuple[int, int]:
    d0 = _parse_day(start_date)
    d1 = _parse_day(end_date)
    start_dt = datetime.combine(d0, time.min, tzinfo=timezone.utc)
    end_dt = datetime.combine(d1, time.max, tzinfo=timezone.utc)
    return int(start_dt.timestamp()), int(end_dt.timestamp())


def _winner_to_settlement_yes(winner: object) -> float:
    if winner is None or (isinstance(winner, float) and pd.isna(winner)):
        return float("nan")
    w = str(winner).strip().lower()
    if w in ("up", "yes"):
        return 1.0
    if w in ("down", "no"):
        return 0.0
    return float("nan")


def _cache_key(
    path: Path,
    *,
    start_date: Optional[str],
    end_date: Optional[str],
    decision_every_sec: int,
    data_start_elapsed: int,
) -> str:
    st = path.stat()
    raw = (
        f"{path.resolve()}|{st.st_mtime}|{st.st_size}|{start_date}|{end_date}|"
        f"{decision_every_sec}|{data_start_elapsed}|nearest_fwd_v1|yes_no_v1|feature_engine_v1"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _normalized_cache_path(s: Settings, key: str) -> Path:
    return cache_root(s) / f"btc_5m_2s_normalized_{key}.parquet"


def _subsample_decision_grid(
    group: pd.DataFrame,
    *,
    min_elapsed: int,
    every: int,
) -> pd.DataFrame:
    """
    For each target elapsed (min_elapsed, min_elapsed+every, ...), keep the first
    row with elapsed >= target (next available tick). Deduplicate by ts.
    """
    g = group.sort_values("elapsed").reset_index(drop=True)
    if every <= 1 or g.empty:
        return g
    targets = list(range(min_elapsed, WINDOW_DURATION_SEC, every))
    picked: list[int] = []
    for target in targets:
        at_or_after = g[g["elapsed"] >= target]
        if at_or_after.empty:
            break
        picked.append(int(at_or_after.index[0]))
    if not picked:
        return g.iloc[0:0]
    return g.loc[picked].drop_duplicates(subset=["ts"], keep="first")


def normalize_btc_5m_replay(
    df: pd.DataFrame,
    *,
    settings: Optional[Settings] = None,
    decision_every_sec: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convert raw btc_5m_2s rows into engine replay schema.

    - Drops elapsed < data_start_elapsed (default 101s blind period).
    - Subsamples trading rows by decision_every_sec: for each grid point, uses the
      next available tick at or after that elapsed (not exact-match only).
    - Appends synthetic expiry row per market at market_end_ts.
    """
    s = settings or get_settings()
    every = decision_every_sec if decision_every_sec is not None else s.replay_decision_every_sec
    min_elapsed = s.btc_5m_data_start_elapsed

    missing = _RAW_COLS - set(df.columns)
    if missing:
        raise ValueError(f"btc_5m_2s parquet missing columns: {sorted(missing)}")

    work = df.copy()
    work["ts"] = work["timestamp_log"].astype(np.int64)
    work["start_time"] = work["start_time"].astype(np.int64)
    work["elapsed"] = work["elapsed"].astype(np.int64)
    work = work[work["elapsed"] >= min_elapsed].copy()

    work["event_slug"] = work["slug"].astype(str)
    work["market_id"] = work["event_slug"]
    work["yes_token_id"] = work["event_slug"]
    work["market_end_ts"] = work["start_time"] + WINDOW_DURATION_SEC
    work["yes_price"] = (work["ask_YES"].astype(float) + work["bid_YES"].astype(float)) / 2.0
    work["no_price"] = (work["ask_NO"].astype(float) + work["bid_NO"].astype(float)) / 2.0
    work["ask_yes"] = work["ask_YES"].astype(float)
    work["bid_yes"] = work["bid_YES"].astype(float)
    work["ask_no"] = work["ask_NO"].astype(float)
    work["bid_no"] = work["bid_NO"].astype(float)
    work["elapsed_sec"] = work["elapsed"]
    work["secs_to_expiry"] = (work["market_end_ts"] - work["ts"]).astype(float)
    work["mins_to_expiry"] = work["secs_to_expiry"] / 60.0
    work["btc_price"] = work["btc_current"].astype(float)
    work["btc_strike"] = work["btc_strike"].astype(float)
    work["btc_gap"] = work["btc_gap"].astype(float)
    # Momentum/vol come from FeatureEngine at runtime; keep columns for schema compat.
    work["btc_ret_5m"] = np.nan
    work["btc_vol_30m"] = np.nan

    settlement_map = (
        work.groupby("event_slug")["winner"]
        .first()
        .map(_winner_to_settlement_yes)
        .to_dict()
    )
    work["settlement_yes"] = work["event_slug"].map(settlement_map)
    work["question"] = None

    by_slug: dict[str, pd.DataFrame] = {}
    for slug, g in work.groupby("event_slug", sort=False):
        by_slug[str(slug)] = g.sort_values("ts")

    if every > 1:
        subsampled = [
            _subsample_decision_grid(by_slug[slug], min_elapsed=min_elapsed, every=every)
            for slug in by_slug
        ]
        work = pd.concat(subsampled, ignore_index=True)
    else:
        work = pd.concat(by_slug.values(), ignore_index=True)

    expiry_rows: list[dict[str, object]] = []
    for slug, g in by_slug.items():
        last = g.sort_values("ts").iloc[-1]
        end_ts = int(last["market_end_ts"])
        if int(last["ts"]) >= end_ts:
            continue
        expiry_rows.append(
            {
                "ts": end_ts,
                "start_time": int(last["start_time"]),
                "event_slug": slug,
                "market_id": slug,
                "yes_token_id": slug,
                "market_end_ts": end_ts,
                "yes_price": float(last["yes_price"]),
                "no_price": float(last["no_price"]),
                "ask_yes": float(last["ask_yes"]),
                "bid_yes": float(last["bid_yes"]),
                "ask_no": float(last["ask_no"]),
                "bid_no": float(last["bid_no"]),
                "elapsed_sec": WINDOW_DURATION_SEC,
                "secs_to_expiry": 0.0,
                "mins_to_expiry": 0.0,
                "btc_price": float(last["btc_price"]),
                "btc_strike": float(last["btc_strike"]),
                "btc_gap": float(last["btc_gap"]),
                "btc_ret_5m": float(last["btc_ret_5m"]) if pd.notna(last["btc_ret_5m"]) else np.nan,
                "btc_vol_30m": float(last["btc_vol_30m"]) if pd.notna(last["btc_vol_30m"]) else np.nan,
                "settlement_yes": settlement_map.get(slug, np.nan),
                "question": None,
            }
        )

    if expiry_rows:
        work = pd.concat([work, pd.DataFrame(expiry_rows)], ignore_index=True)

    out_cols = [
        "ts",
        "market_id",
        "yes_token_id",
        "event_slug",
        "yes_price",
        "no_price",
        "ask_yes",
        "bid_yes",
        "ask_no",
        "bid_no",
        "elapsed_sec",
        "secs_to_expiry",
        "mins_to_expiry",
        "btc_price",
        "btc_strike",
        "btc_gap",
        "btc_ret_5m",
        "btc_vol_30m",
        "market_end_ts",
        "settlement_yes",
        "question",
    ]
    out = work[out_cols].sort_values(["ts", "market_id"]).reset_index(drop=True)
    return out


def load_btc_5m_replay(
    path: Optional[Path] = None,
    *,
    settings: Optional[Settings] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    decision_every_sec: Optional[int] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load btc_5m_2s parquet/csv, optionally filter by UTC date on start_time, normalize."""
    s = settings or get_settings()
    src = path or s.btc_5m_replay_path
    if not src.is_absolute():
        src = s.project_root / src
    if not src.exists():
        raise FileNotFoundError(
            f"btc_5m replay not found at {src}. Place btc_5m_2s.parquet under data/cache/.",
        )

    every = decision_every_sec if decision_every_sec is not None else s.replay_decision_every_sec
    cache_key = _cache_key(
        src,
        start_date=start_date,
        end_date=end_date,
        decision_every_sec=every,
        data_start_elapsed=s.btc_5m_data_start_elapsed,
    )
    cache_path = _normalized_cache_path(s, cache_key)
    if use_cache and cache_path.exists():
        return read_parquet(cache_path)

    if src.suffix.lower() == ".csv":
        raw = pd.read_csv(src)
    else:
        raw = pd.read_parquet(src)

    if start_date and end_date:
        lo, hi = _utc_range_ts(start_date, end_date)
        raw = raw[(raw["start_time"] >= lo) & (raw["start_time"] <= hi)].copy()
    elif start_date:
        lo, _ = _utc_range_ts(start_date, start_date)
        raw = raw[raw["start_time"] >= lo].copy()
    elif end_date:
        _, hi = _utc_range_ts(end_date, end_date)
        raw = raw[raw["start_time"] <= hi].copy()

    if raw.empty:
        raise ValueError("No btc_5m_2s rows after date filter.")

    normalized = normalize_btc_5m_replay(raw, settings=s, decision_every_sec=every)
    if use_cache:
        write_parquet(normalized, cache_path)
    return normalized
