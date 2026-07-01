"""Agent-specific precomputed feature payloads."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

WINDOW_SEC = 300.0
OBSERVABLE_START_ELAPSED = 101


class MarketDynamicsFeatures(BaseModel):
    """Market dynamics agent — BTC barrier state, momentum, vol."""

    secs_to_expiry: Optional[float] = None
    elapsed_pct: Optional[float] = None
    observable_elapsed: Optional[float] = None

    strike_gap_pct: Optional[float] = None
    gap_abs_pct: Optional[float] = None
    gap_change_30s: Optional[float] = None
    gap_change_60s: Optional[float] = None
    gap_zero_crossings: Optional[int] = None
    gap_range_obs: Optional[float] = None

    return_30s: Optional[float] = None
    return_60s: Optional[float] = None
    trend_slope_60s: Optional[float] = None
    momentum_accel: Optional[float] = None
    btc_range_60s: Optional[float] = None

    vol_60s: Optional[float] = None
    vol_120s: Optional[float] = None
    vol_ratio_60_120: Optional[float] = None
    vol_scaled_gap: Optional[float] = None

    model_prob_up: Optional[float] = None
    pin_risk: Optional[float] = None
    barrier_certainty: Optional[float] = None


class PredictionMarketFeatures(BaseModel):
    """Prediction market agent — Polymarket microstructure & cross-signal agreement."""

    yes_mid: float = 0.5
    yes_spread_pct: Optional[float] = None
    prob_divergence: float = 0.0
    yes_mid_change_30s: Optional[float] = None
    yes_mid_change_60s: Optional[float] = None
    repricing_beta_30s: Optional[float] = None
    repricing_lag: Optional[float] = None
    gap_poly_agree: Optional[float] = None
    model_vs_market: Optional[float] = None
    signal_dispersion: Optional[float] = None
    consensus_strength: Optional[float] = None
    half_spread_cost: Optional[float] = None
    model_prob_up: Optional[float] = None


# Legacy aliases
BtcFeatures = MarketDynamicsFeatures
PolyFeatures = PredictionMarketFeatures

__all__ = [
    "MarketDynamicsFeatures",
    "PredictionMarketFeatures",
    "BtcFeatures",
    "PolyFeatures",
    "WINDOW_SEC",
    "OBSERVABLE_START_ELAPSED",
]
