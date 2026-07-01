"""Unit checks for FeatureEngine and caches."""

from __future__ import annotations

from btc5m_agents.features import (
    FeatureEngine,
    MarketDynamicsFeatures,
    MarketStateCache,
    PredictionMarketFeatures,
)
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


def test_split_agent_feature_payloads() -> None:
    cache = MarketStateCache(history_sec=900)
    engine = FeatureEngine()
    prices = [100_000.0, 100_010.0, 100_020.0, 100_030.0]
    dynamics: MarketDynamicsFeatures | None = None
    prediction: PredictionMarketFeatures | None = None
    for i, p in enumerate(prices):
        snap = _snap(1000 + i * 30, p, yes=0.48 + i * 0.01)
        cache.update(snap)
        dynamics, prediction = engine.compute(cache, snap)

    assert dynamics is not None and prediction is not None
    assert dynamics.return_60s is not None
    assert dynamics.return_60s > 0
    assert dynamics.model_prob_up is not None
    assert prediction.model_prob_up == dynamics.model_prob_up
    assert prediction.yes_spread_pct is not None
    assert abs(prediction.yes_spread_pct - 0.02 / prediction.yes_mid) < 0.01
    assert dynamics.observable_elapsed == 19.0
    assert prediction.half_spread_cost == round(prediction.yes_spread_pct / 2, 4)
    assert set(dynamics.model_dump()) == set(MarketDynamicsFeatures.model_fields)
    assert set(prediction.model_dump()) == set(PredictionMarketFeatures.model_fields)
    assert "yes_mid" not in dynamics.model_dump()
    assert "strike_gap_pct" not in prediction.model_dump()
