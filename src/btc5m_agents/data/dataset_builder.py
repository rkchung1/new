"""Build aligned 1-minute replay Parquet from markets + CLOB + BTC candles.

Dormant: not used by run_backtest. Retained for possible future API-based replay.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from btc5m_agents.config import Settings, get_settings
from btc5m_agents.data.btc_price_client import BtcPriceClient
from btc5m_agents.data.cache import cache_root, read_parquet, write_parquet
from btc5m_agents.data.market_discovery import load_markets_bundle, window_ts_from_slug
from btc5m_agents.data.polymarket_client import PolymarketClient


def _minute_grid(start_ts: int, end_ts: int, step: int = 60) -> np.ndarray:
    """Timestamps in [start_ts, end_ts) aligned to `step` seconds."""
    t0 = int(math.ceil(start_ts / step) * step)
    out: list[int] = []
    t = t0
    while t < end_ts:
        out.append(t)
        t += step
    return np.array(out, dtype=np.int64)


def _replay_ts_grid(start_ts: int, end_ts: int, step: int = 60) -> np.ndarray:
    """Like `_minute_grid` but includes `end_ts` so expiry row gets fresh BTC/YES joins."""
    grid = _minute_grid(start_ts, end_ts, step)
    if len(grid) == 0:
        return np.array([np.int64(end_ts)], dtype=np.int64)
    if int(grid[-1]) < end_ts:
        grid = np.append(grid, np.int64(end_ts))
    return grid


def _clob_cache_path(s: Settings, token_id: str, start_ts: int, end_ts: int) -> Path:
    safe = f"{token_id}_{start_ts}_{end_ts}.parquet"
    return cache_root(s) / "poly" / safe


def _fetch_or_load_clob(
    client: PolymarketClient,
    s: Settings,
    token_id: str,
    start_ts: int,
    end_ts: int,
) -> pd.DataFrame:
    path = _clob_cache_path(s, token_id, start_ts, end_ts)
    if path.exists():
        return read_parquet(path)
    hist = client.get_prices_history(token_id, start_ts, end_ts + 60, fidelity=1, interval="max")
    if not hist:
        df = pd.DataFrame(columns=["ts", "yes_price"])
        write_parquet(df, path)
        return df
    rows = []
    for row in hist:
        if isinstance(row, dict) and "t" in row and "p" in row:
            rows.append({"ts": int(row["t"]), "yes_price": float(row["p"])})
    df = pd.DataFrame(rows).sort_values("ts").drop_duplicates(subset=["ts"])
    write_parquet(df, path)
    return df


def _build_btc_minute_features(
    s: Settings,
    global_start: int,
    global_end: int,
) -> pd.DataFrame:
    # Cache key includes "open" — btc_price uses candle open (bucket start), not close.
    cache_path = cache_root(s) / "btc" / f"btc_1m_open_{global_start}_{global_end}.parquet"
    if cache_path.exists():
        return read_parquet(cache_path)
    client = BtcPriceClient(s)
    raw = client.fetch_candles(global_start - 3600, global_end + 3600)
    if not raw:
        empty = pd.DataFrame(
            columns=["ts", "btc_price", "btc_ret_5m", "btc_vol_30m"],
        )
        write_parquet(empty, cache_path)
        return empty
    c1m = pd.DataFrame(
        raw,
        columns=["ts", "low", "high", "open", "close", "volume"],
    ).sort_values("ts")
    c1m["ts"] = c1m["ts"].astype(np.int64)
    c1m["open"] = c1m["open"].astype(float)

    minute_ts = _minute_grid(global_start - 1800, global_end + 1800, 60)
    base = pd.DataFrame({"ts": minute_ts})
    merged = pd.merge_asof(
        base.sort_values("ts"),
        c1m[["ts", "open"]].sort_values("ts"),
        on="ts",
        direction="backward",
    )
    merged.rename(columns={"open": "btc_price"}, inplace=True)
    merged["btc_price"] = merged["btc_price"].ffill()
    merged["btc_ret_5m"] = merged["btc_price"] / merged["btc_price"].shift(5) - 1.0
    merged["btc_vol_30m"] = merged["btc_price"].pct_change().rolling(30).std()
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    out = merged[["ts", "btc_price", "btc_ret_5m", "btc_vol_30m"]]
    write_parquet(out, cache_path)
    return out


def build_replay_dataset(
    start_date: str,
    end_date: str,
    *,
    settings: Optional[Settings] = None,
) -> Path:
    """
    Produce `replay_{start_date}_{end_date}.parquet` with 1 row per (minute, market).

    Each market's window is `[slug_unix, slug_unix + 300)` from `event_slug`, not Gamma dates.
    Columns include `settlement_yes` for engine-only settlement (not passed to agents).
    """
    s = settings or get_settings()
    markets = load_markets_bundle(start_date, end_date, s)
    if markets.empty:
        raise FileNotFoundError(
            f"No markets bundle at markets_{start_date}_{end_date}.parquet; run discover_markets first.",
        )

    windows: list[tuple[int, int]] = []
    for slug in markets["event_slug"]:
        w = window_ts_from_slug(str(slug), prefix=s.slug_prefix)
        if w is not None:
            windows.append(w)
    if not windows:
        raise ValueError("No markets with parseable slug windows in bundle.")
    global_start = min(w[0] for w in windows) - 3600
    global_end = max(w[1] for w in windows) + 3600
    btc_min = _build_btc_minute_features(s, global_start, global_end)

    clob = PolymarketClient(s)
    frames: list[pd.DataFrame] = []

    for _, row in markets.iterrows():
        window = window_ts_from_slug(str(row["event_slug"]), prefix=s.slug_prefix)
        if window is None:
            continue
        m_start, m_end = window
        yes_token = str(row["yes_token_id"])
        hist_df = _fetch_or_load_clob(clob, s, yes_token, m_start, m_end)
        grid = _replay_ts_grid(m_start, m_end, 60)
        gdf = pd.DataFrame({"ts": grid})
        if hist_df.empty:
            gdf["yes_price"] = np.nan
        else:
            gdf = pd.merge_asof(
                gdf.sort_values("ts"),
                hist_df.sort_values("ts"),
                on="ts",
                direction="backward",
            )
            gdf["yes_price"] = gdf["yes_price"].ffill()

        if not btc_min.empty:
            gdf = pd.merge_asof(
                gdf.sort_values("ts"),
                btc_min.sort_values("ts"),
                on="ts",
                direction="backward",
            )
        else:
            gdf["btc_price"] = np.nan
            gdf["btc_ret_5m"] = np.nan
            gdf["btc_vol_30m"] = np.nan

        gdf["market_id"] = str(row["market_id"])
        gdf["yes_token_id"] = yes_token
        gdf["event_slug"] = str(row["event_slug"])
        gdf["market_end_ts"] = m_end
        gdf["mins_to_expiry"] = (m_end - gdf["ts"]) / 60.0
        gdf["question"] = row.get("question")
        sy = row.get("settlement_yes")
        gdf["settlement_yes"] = float(sy) if sy is not None and not pd.isna(sy) else np.nan
        frames.append(gdf)

    if not frames:
        replay = pd.DataFrame(
            columns=[
                "ts",
                "market_id",
                "yes_token_id",
                "event_slug",
                "yes_price",
                "mins_to_expiry",
                "btc_price",
                "btc_ret_5m",
                "btc_vol_30m",
                "market_end_ts",
                "settlement_yes",
                "question",
            ],
        )
    else:
        replay = pd.concat(frames, ignore_index=True)
        replay.sort_values(["ts", "market_id"], inplace=True)
        replay.reset_index(drop=True, inplace=True)

    out_path = cache_root(s) / f"replay_{start_date}_{end_date}.parquet"
    write_parquet(replay, out_path)
    return out_path
