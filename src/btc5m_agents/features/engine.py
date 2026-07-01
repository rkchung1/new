"""Compute agent-specific feature payloads from rolling caches."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from btc5m_agents.features.cache import BookTick, MarketStateCache
from btc5m_agents.features.models import (
    OBSERVABLE_START_ELAPSED,
    WINDOW_SEC,
    MarketDynamicsFeatures,
    PredictionMarketFeatures,
)
from btc5m_agents.types import Snapshot

_VOL_FLOOR = 1e-8
_GAP_EPS = 1e-7
_YES_CHG_LAG_EPS = 0.001
_GAP_CHG_LAG_EPS = 1e-6


def _round4(v: Optional[float]) -> Optional[float]:
    if v is None or v != v:
        return None
    return float(round(v, 4))


def _safe_return(current: float, past: Optional[float]) -> Optional[float]:
    if past is None or past != past or current != current or past <= 0:
        return None
    return current / past - 1.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _sign_scalar(v: Optional[float], eps: float = 1e-12) -> int:
    if v is None or v != v or abs(v) <= eps:
        return 0
    return 1 if v > 0 else -1


def _rolling_vol(points: list, lag_sec: int, current_ts: int) -> Optional[float]:
    """Std of log returns over points in [current_ts - lag_sec, current_ts]."""
    prices = [p.price for p in points if current_ts - lag_sec <= p.ts <= current_ts]
    if len(prices) < 3:
        return None
    rets = np.diff(np.log(np.maximum(prices, 1e-12)))
    if len(rets) < 2:
        return None
    return float(np.std(rets))


def _trend_slope(points: list, lag_sec: int, current_ts: int) -> Optional[float]:
    subset = [(p.ts, p.price) for p in points if current_ts - lag_sec <= p.ts <= current_ts]
    if len(subset) < 3:
        return None
    xs = np.array([t - subset[0][0] for t, _ in subset], dtype=float)
    ys = np.array([p for _, p in subset], dtype=float)
    if np.std(xs) < 1e-9:
        return None
    return float(np.polyfit(xs, ys, 1)[0])


def _btc_range(points: list, lag_sec: int, current_ts: int) -> Optional[float]:
    prices = [p.price for p in points if current_ts - lag_sec <= p.ts <= current_ts]
    if len(prices) < 2:
        return None
    lo, hi = min(prices), max(prices)
    mid = (lo + hi) / 2.0
    if mid <= 0:
        return None
    return (hi - lo) / mid


def _spread(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None or bid != bid or ask != ask:
        return None
    return float(ask - bid)


def _gap_pct_from_tick(tick: BookTick) -> Optional[float]:
    gap, strike = tick.btc_gap, tick.btc_strike
    if gap is None or strike is None or gap != gap or strike <= 0:
        return None
    return gap / strike


def _observable_gap_pcts(ticks: list[BookTick]) -> list[float]:
    out: list[float] = []
    for t in ticks:
        if t.elapsed_sec is not None and t.elapsed_sec < OBSERVABLE_START_ELAPSED:
            continue
        g = _gap_pct_from_tick(t)
        if g is not None:
            out.append(g)
    return out


def _gap_path_stats(ticks: list[BookTick]) -> tuple[Optional[float], Optional[int]]:
    gaps = _observable_gap_pcts(ticks)
    if not gaps:
        return None, None
    gap_range = max(gaps) - min(gaps)
    crossings = 0
    prev_sign: Optional[int] = None
    for g in gaps:
        sign = _sign_scalar(g, _GAP_EPS)
        if prev_sign is not None and sign != 0 and prev_sign != 0 and sign != prev_sign:
            crossings += 1
        if sign != 0:
            prev_sign = sign
    return gap_range, crossings


def _consensus_metrics(
    gap_pct: Optional[float],
    ret_60: Optional[float],
    slope: Optional[float],
    yes_chg_30: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    signs = [
        _sign_scalar(gap_pct, _GAP_EPS),
        _sign_scalar(ret_60),
        _sign_scalar(slope),
        _sign_scalar(yes_chg_30),
    ]
    active = [s for s in signs if s != 0]
    if not active:
        return None, None
    n_up = sum(1 for s in active if s > 0)
    n_down = len(active) - n_up
    strength = abs(n_up - n_down) / len(active)
    dispersion = 1.0 - strength
    return _round4(dispersion), _round4(strength)


def _barrier_probs(
    gap_pct: Optional[float],
    vol: Optional[float],
    secs_to_expiry: Optional[float],
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if gap_pct is None or gap_pct != gap_pct:
        return None, None, None, None
    if secs_to_expiry is None or secs_to_expiry <= 0:
        return None, None, None, None
    vol_use = vol if vol is not None and vol == vol and vol > _VOL_FLOOR else _VOL_FLOOR
    denom = vol_use * math.sqrt(secs_to_expiry)
    if denom <= 0:
        return None, None, None, None
    z = gap_pct / denom
    model_up = _norm_cdf(z)
    model_up = float(max(0.0, min(1.0, model_up)))
    pin = math.exp(-abs(z)) * (1.0 - secs_to_expiry / WINDOW_SEC)
    certainty = abs(model_up - 0.5) * 2.0
    return _round4(z), _round4(model_up), _round4(pin), _round4(certainty)


class FeatureEngine:
    def compute(
        self,
        cache: MarketStateCache,
        snap: Snapshot,
    ) -> tuple[MarketDynamicsFeatures, PredictionMarketFeatures]:
        tape = cache.btc_tape.series(snap.ts)
        price = snap.btc_price if snap.btc_price == snap.btc_price else float("nan")
        strike = snap.btc_strike
        gap = snap.btc_gap
        gap_pct = (
            (gap / strike)
            if gap is not None and strike is not None and strike > 0 and gap == gap
            else None
        )
        gap_abs = abs(gap_pct) if gap_pct is not None else None

        book = cache.book(snap.market_id)
        book_ticks = book.ticks(snap.ts)

        gap_30 = book.value_at_lag(snap.ts, 30, "btc_gap")
        gap_60 = book.value_at_lag(snap.ts, 60, "btc_gap")
        gap_chg_30 = (
            (gap - gap_30)
            if gap is not None and gap_30 is not None and gap == gap and gap_30 == gap_30
            else None
        )
        gap_chg_60 = (
            (gap - gap_60)
            if gap is not None and gap_60 is not None and gap == gap and gap_60 == gap_60
            else None
        )
        gap_pct_30 = (
            (gap_30 / strike)
            if gap_30 is not None and strike is not None and strike > 0 and gap_30 == gap_30
            else None
        )
        gap_pct_chg_30 = (
            (gap_pct - gap_pct_30)
            if gap_pct is not None and gap_pct_30 is not None
            else None
        )

        gap_range_obs, gap_crossings = _gap_path_stats(book_ticks)

        ret_30 = _safe_return(price, cache.btc_tape.price_at_lag(snap.ts, 30))
        ret_60 = _safe_return(price, cache.btc_tape.price_at_lag(snap.ts, 60))
        ret_30_r = _round4(ret_30)
        ret_60_r = _round4(ret_60)
        mom_accel = (
            ret_30 - ret_60 / 2.0
            if ret_30 is not None and ret_60 is not None
            else None
        )

        vol_60 = _round4(_rolling_vol(tape, 60, snap.ts))
        vol_120 = _round4(_rolling_vol(tape, 120, snap.ts))
        vol_ratio = (
            vol_60 / vol_120
            if vol_60 is not None and vol_120 is not None and vol_120 > _VOL_FLOOR
            else None
        )

        vol_scaled, model_up, pin_risk, barrier_cert = _barrier_probs(
            gap_pct, vol_120, snap.secs_to_expiry
        )

        yes_mid = float(snap.yes_price) if snap.yes_price == snap.yes_price else 0.5
        no_mid = float(snap.no_price) if snap.no_price == snap.no_price else (1.0 - yes_mid)

        y_spread = _spread(snap.bid_yes, snap.ask_yes)
        y_spread_pct = (y_spread / yes_mid) if y_spread is not None and yes_mid > 0 else None
        half_spread = (y_spread_pct / 2.0) if y_spread_pct is not None else None

        yes_30 = book.value_at_lag(snap.ts, 30, "yes_mid")
        yes_60 = book.value_at_lag(snap.ts, 60, "yes_mid")
        yes_chg_30 = (yes_mid - yes_30) if yes_30 is not None and yes_30 == yes_30 else None
        yes_chg_60 = (yes_mid - yes_60) if yes_60 is not None and yes_60 == yes_60 else None

        repricing_beta = (
            yes_chg_30 / gap_pct_chg_30
            if yes_chg_30 is not None
            and gap_pct_chg_30 is not None
            and abs(gap_pct_chg_30) > _GAP_CHG_LAG_EPS
            else None
        )
        repricing_lag = (
            1.0
            if gap_pct_chg_30 is not None
            and abs(gap_pct_chg_30) > _GAP_CHG_LAG_EPS
            and yes_chg_30 is not None
            and abs(yes_chg_30) < _YES_CHG_LAG_EPS
            else 0.0
        )

        gap_poly_agree = None
        if gap_pct is not None:
            gap_sign = _sign_scalar(gap_pct, _GAP_EPS)
            poly_sign = _sign_scalar(yes_mid - 0.5, 0.01)
            if gap_sign != 0 and poly_sign != 0:
                gap_poly_agree = 1.0 if gap_sign == poly_sign else 0.0

        model_vs_mkt = (
            model_up - yes_mid
            if model_up is not None
            else None
        )

        signal_disp, consensus = _consensus_metrics(
            gap_pct, ret_60, _trend_slope(tape, 60, snap.ts), yes_chg_30
        )

        elapsed = float(snap.elapsed_sec) if snap.elapsed_sec is not None else None
        elapsed_pct = (elapsed / WINDOW_SEC) if elapsed is not None else None
        obs_elapsed = (
            max(0.0, elapsed - OBSERVABLE_START_ELAPSED)
            if elapsed is not None
            else None
        )

        dynamics = MarketDynamicsFeatures(
            secs_to_expiry=_round4(snap.secs_to_expiry),
            elapsed_pct=_round4(elapsed_pct),
            observable_elapsed=_round4(obs_elapsed),
            strike_gap_pct=_round4(gap_pct),
            gap_abs_pct=_round4(gap_abs),
            gap_change_30s=_round4(gap_chg_30),
            gap_change_60s=_round4(gap_chg_60),
            gap_zero_crossings=gap_crossings,
            gap_range_obs=_round4(gap_range_obs),
            return_30s=ret_30_r,
            return_60s=ret_60_r,
            trend_slope_60s=_round4(_trend_slope(tape, 60, snap.ts)),
            momentum_accel=_round4(mom_accel),
            btc_range_60s=_round4(_btc_range(tape, 60, snap.ts)),
            vol_60s=vol_60,
            vol_120s=vol_120,
            vol_ratio_60_120=_round4(vol_ratio),
            vol_scaled_gap=vol_scaled,
            model_prob_up=model_up,
            pin_risk=pin_risk,
            barrier_certainty=barrier_cert,
        )
        prediction = PredictionMarketFeatures(
            yes_mid=_round4(yes_mid) or yes_mid,
            yes_spread_pct=_round4(y_spread_pct),
            prob_divergence=_round4(yes_mid + no_mid - 1.0) or 0.0,
            yes_mid_change_30s=_round4(yes_chg_30),
            yes_mid_change_60s=_round4(yes_chg_60),
            repricing_beta_30s=_round4(repricing_beta),
            repricing_lag=repricing_lag,
            gap_poly_agree=gap_poly_agree,
            model_vs_market=_round4(model_vs_mkt),
            signal_dispersion=signal_disp,
            consensus_strength=consensus,
            half_spread_cost=_round4(half_spread),
            model_prob_up=model_up,
        )
        return dynamics, prediction
