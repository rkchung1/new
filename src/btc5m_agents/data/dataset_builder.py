"""Build aligned 1-minute replay Parquet from markets + CLOB + BTC candles."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from btc5m_agents.config import Settings, get_settings
from btc5m_agents.data.btc_price_client import BtcPriceClient
from btc5m_agents.data.cache import cache_root, read_parquet, write_parquet
from btc5m_agents.data.market_discovery import load_markets_bundle
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
    cache_path = cache_root(s) / "btc" / f"btc_minute_{global_start}_{global_end}.parquet"
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
    c5 = pd.DataFrame(
        raw,
        columns=["ts", "low", "high", "open", "close", "volume"],
    ).sort_values("ts")
    c5["ts"] = c5["ts"].astype(np.int64)
    c5["close"] = c5["close"].astype(float)

    minute_ts = _minute_grid(global_start - 1800, global_end + 1800, 60)
    base = pd.DataFrame({"ts": minute_ts})
    merged = pd.merge_asof(
        base.sort_values("ts"),
        c5[["ts", "close"]].sort_values("ts"),
        on="ts",
        direction="backward",
    )
    merged.rename(columns={"close": "btc_price"}, inplace=True)
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

    Columns include `settlement_yes` for engine-only settlement (not passed to agents).
    """
    s = settings or get_settings()
    markets = load_markets_bundle(start_date, end_date, s)
    if markets.empty:
        raise FileNotFoundError(
            f"No markets bundle at markets_{start_date}_{end_date}.parquet; run discover_markets first.",
        )

    global_start = int(markets["start_ts"].min()) - 3600
    global_end = int(markets["end_ts"].max()) + 3600
    btc_min = _build_btc_minute_features(s, global_start, global_end)

    clob = PolymarketClient(s)
    frames: list[pd.DataFrame] = []

    for _, row in markets.iterrows():
        m_start = int(row["start_ts"])
        m_end = int(row["end_ts"])
        yes_token = str(row["yes_token_id"])
        hist_df = _fetch_or_load_clob(clob, s, yes_token, m_start, m_end)
        grid = _minute_grid(m_start, m_end, 60)
        if len(grid) == 0:
            continue
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
        # Terminal snapshot at exact expiry so the engine can settle with ts >= market_end_ts
        # Avoid DataFrame([Series]) + .loc on row 0: it upcasts dtypes and breaks concat on pandas 2.3+.
        if not gdf.empty and int(gdf["ts"].max()) < m_end:
            terminal = gdf.iloc[-1:].copy()
            terminal["ts"] = np.int64(m_end)
            terminal["mins_to_expiry"] = 0.0
            gdf = pd.concat([gdf, terminal], ignore_index=True)
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
