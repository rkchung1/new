"""Compact feature vectors for analyst agents."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class BtcFeatures(BaseModel):
    """Price Analyst input — compressed directional BTC state."""

    return_60s: Optional[float] = None
    return_300s: Optional[float] = None
    vol_120s: Optional[float] = None
    trend_slope_60s: Optional[float] = None
    strike_gap_pct: Optional[float] = None
    gap_change_30s: Optional[float] = None
    secs_to_expiry: Optional[float] = None


class PolyFeatures(BaseModel):
    """Polymarket Analyst input — book state plus compact BTC context."""

    yes_mid: float
    yes_spread_pct: Optional[float] = None
    yes_imbalance: Optional[float] = None
    prob_divergence: float
    yes_mid_change_30s: Optional[float] = None
    strike_gap_pct: Optional[float] = None
    return_60s: Optional[float] = None
    vol_120s: Optional[float] = None
    secs_to_expiry: Optional[float] = None
