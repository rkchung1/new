"""Compute BtcFeatures and PolyFeatures from rolling caches."""

from __future__ import annotations

from typing import Optional

import numpy as np

from btc5m_agents.features.cache import MarketStateCache
from btc5m_agents.features.models import BtcFeatures, PolyFeatures
from btc5m_agents.types import Snapshot

def _round4(v: Optional[float]) -> Optional[float]:
    if v is None or v != v:
        return None
    return float(round(v, 4))


def _safe_return(current: float, past: Optional[float]) -> Optional[float]:
    if past is None or past != past or current != current or past <= 0:
        return None
    return current / past - 1.0


def _rolling_vol(points: list, lag_sec: int, current_ts: int) -> Optional[float]:
    """Std of log returns over points in (current_ts - lag_sec, current_ts]."""
    if len(points) < 3:
        return None
    prices = [p.price for p in points if p.ts <= current_ts]
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
    slope = np.polyfit(xs, ys, 1)[0]
    return float(slope)


def _spread(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None or bid != bid or ask != ask:
        return None
    return float(ask - bid)


def _imbalance(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None or bid != bid or ask != ask:
        return None
    total = bid + ask
    if total <= 0:
        return None
    return float((bid - ask) / total)


class FeatureEngine:
    def compute(
        self,
        cache: MarketStateCache,
        snap: Snapshot,
    ) -> tuple[BtcFeatures, PolyFeatures]:
        btc = self._btc_features(cache, snap)
        poly = self._poly_features(cache, snap, btc)
        return btc, poly

    def _btc_features(self, cache: MarketStateCache, snap: Snapshot) -> BtcFeatures:
        tape = cache.btc_tape.series(snap.ts)
        price = snap.btc_price if snap.btc_price == snap.btc_price else float("nan")
        strike = snap.btc_strike
        gap = snap.btc_gap
        gap_pct = (gap / strike) if gap is not None and strike is not None and strike > 0 and gap == gap else None

        book = cache.book(snap.market_id)
        gap_30 = book.value_at_lag(snap.ts, 30, "btc_gap")
        gap_chg = (gap - gap_30) if gap is not None and gap_30 is not None and gap == gap and gap_30 == gap_30 else None

        return BtcFeatures(
            return_60s=_round4(_safe_return(price, cache.btc_tape.price_at_lag(snap.ts, 60))),
            return_300s=_round4(_safe_return(price, cache.btc_tape.price_at_lag(snap.ts, 300))),
            vol_120s=_round4(_rolling_vol(tape, 120, snap.ts)),
            trend_slope_60s=_round4(_trend_slope(tape, 60, snap.ts)),
            strike_gap_pct=_round4(gap_pct),
            gap_change_30s=_round4(gap_chg),
            secs_to_expiry=_round4(snap.secs_to_expiry),
        )

    def _poly_features(
        self,
        cache: MarketStateCache,
        snap: Snapshot,
        btc: BtcFeatures,
    ) -> PolyFeatures:
        yes_mid = float(snap.yes_price) if snap.yes_price == snap.yes_price else 0.5
        no_mid = float(snap.no_price) if snap.no_price == snap.no_price else (1.0 - yes_mid)

        y_spread = _spread(snap.bid_yes, snap.ask_yes)
        y_spread_pct = (y_spread / yes_mid) if y_spread is not None and yes_mid > 0 else None

        book = cache.book(snap.market_id)
        yes_30 = book.value_at_lag(snap.ts, 30, "yes_mid")
        yes_chg = (yes_mid - yes_30) if yes_30 is not None and yes_30 == yes_30 else None

        return PolyFeatures(
            yes_mid=_round4(yes_mid) or yes_mid,
            yes_spread_pct=_round4(y_spread_pct),
            yes_imbalance=_round4(_imbalance(snap.bid_yes, snap.ask_yes)),
            prob_divergence=_round4(yes_mid + no_mid - 1.0) or 0.0,
            yes_mid_change_30s=_round4(yes_chg),
            strike_gap_pct=btc.strike_gap_pct,
            return_60s=btc.return_60s,
            vol_120s=btc.vol_120s,
            secs_to_expiry=_round4(snap.secs_to_expiry),
        )
