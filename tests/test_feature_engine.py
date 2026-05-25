"""Unit checks for FeatureEngine and caches."""

from __future__ import annotations

from btc5m_agents.features import FeatureEngine, MarketStateCache
from btc5m_agents.types import Snapshot


def _snap(ts: int, price: float, yes: float = 0.5, gap: float = 10.0) -> Snapshot:
    return Snapshot(
        ts=ts,
        market_id="m1",
        yes_token_id="t1",
        event_slug="e1",
        yes_price=yes,
        no_price=1.0 - yes,
        mins_to_expiry=4.0,
        btc_price=price,
        btc_ret_5m=float("nan"),
        btc_vol_30m=float("nan"),
        elapsed_sec=120,
        secs_to_expiry=180.0,
        bid_yes=yes - 0.01,
        ask_yes=yes + 0.01,
        bid_no=(1.0 - yes) - 0.01,
        ask_no=(1.0 - yes) + 0.01,
        btc_strike=100_000.0,
        btc_gap=gap,
    )


def test_momentum_and_poly_features() -> None:
    cache = MarketStateCache(history_sec=900)
    engine = FeatureEngine()
    prices = [100_000.0, 100_010.0, 100_020.0, 100_030.0]
    for i, p in enumerate(prices):
        snap = _snap(1000 + i * 30, p, yes=0.48 + i * 0.01)
        cache.update(snap)
        btc_f, poly_f = engine.compute(cache, snap)

    assert btc_f.return_60s is not None
    assert btc_f.return_60s > 0
    assert "btc_price" not in btc_f.model_dump()
    assert "market_id" not in btc_f.model_dump()
    assert poly_f.yes_spread_pct is not None
    assert abs(poly_f.yes_spread_pct - 0.02 / poly_f.yes_mid) < 0.01
    assert "no_mid" not in poly_f.model_dump()
    assert poly_f.strike_gap_pct == btc_f.strike_gap_pct
    assert poly_f.return_60s == btc_f.return_60s
    assert poly_f.vol_120s == btc_f.vol_120s
