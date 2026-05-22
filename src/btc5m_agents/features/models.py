"""Compact feature vectors for analyst agents."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class BtcFeatures(BaseModel):
    """Price Analyst input — BTC momentum, vol, trend, strike, timing."""

    market_id: str
    ts: int
    btc_price: float
    return_30s: Optional[float] = None
    return_60s: Optional[float] = None
    return_120s: Optional[float] = None
    return_300s: Optional[float] = None
    vol_60s: Optional[float] = None
    vol_120s: Optional[float] = None
    trend_slope_60s: Optional[float] = None
    strike_gap: Optional[float] = None
    strike_gap_pct: Optional[float] = None
    gap_change_30s: Optional[float] = None
    elapsed_sec: Optional[int] = None
    secs_to_expiry: Optional[float] = None
    window_pct: Optional[float] = None


class PolyFeatures(BaseModel):
    """Polymarket Analyst input — book, spread, imbalance, inefficiency."""

    market_id: str
    ts: int
    yes_mid: float
    no_mid: float
    mid_sum: float
    yes_spread: Optional[float] = None
    no_spread: Optional[float] = None
    yes_spread_pct: Optional[float] = None
    no_spread_pct: Optional[float] = None
    yes_imbalance: Optional[float] = None
    no_imbalance: Optional[float] = None
    prob_divergence: float
    yes_mid_change_30s: Optional[float] = None
    no_mid_change_30s: Optional[float] = None
    elapsed_sec: Optional[int] = None
    secs_to_expiry: Optional[float] = None
