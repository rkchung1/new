"""Unit checks for risk execution state and sizing helpers."""

from __future__ import annotations

import json

from btc5m_agents.agents.execution import build_risk_execution_state, risk_payload
from btc5m_agents.agents.pricing import compute_edge
from btc5m_agents.agents.sizing import (
    buy_headroom_usd,
    market_exposure_usd,
    sell_caps,
)
from btc5m_agents.config import Settings
from btc5m_agents.types import MarketPosition, PortfolioSnapshot


def _settings() -> Settings:
    return Settings(
        initial_cash=1000.0,
        max_position_per_market_usd=100.0,
        max_total_exposure_usd=300.0,
    )


def _state(
    *,
    cash: float = 1000.0,
    exposure_usd: float = 0.0,
    yes_sh: float = 0.0,
    no_sh: float = 0.0,
    yes_mid: float = 0.5,
    market_id: str = "m1",
) -> dict:
    positions = {}
    if yes_sh > 0 or no_sh > 0:
        positions[market_id] = MarketPosition(
            market_id=market_id,
            yes_shares=yes_sh,
            no_shares=no_sh,
        )
    return {
        "market_id": market_id,
        "portfolio": PortfolioSnapshot(
            cash=cash,
            positions=positions,
            exposure_usd=exposure_usd,
        ).model_dump(mode="json"),
        "price_view": {"direction": "UP", "confidence": 0.7},
        "poly_view": {"fair_prob_up": 0.6, "edge": 0.05},
        "poly_features": {"yes_mid": yes_mid},
        "btc_features": {"secs_to_expiry": 120.0},
    }


def test_empty_position_sell_caps_zero() -> None:
    port = PortfolioSnapshot(cash=1000.0, exposure_usd=0.0)
    s = _settings()
    assert sell_caps(0.0, 0.0, 0.5) == (0.0, 0.0)
    assert buy_headroom_usd(port, "m1", 0.5, s) == 100.0  # min(100-0, 300-0, 250)


def test_long_yes_sell_cap_matches_shares_times_mid() -> None:
    sell_yes, sell_no = sell_caps(12.5, 0.0, 0.4)
    assert sell_yes == 5.0
    assert sell_no == 0.0


def test_buy_cap_shrinks_near_total_exposure() -> None:
    port = PortfolioSnapshot(cash=500.0, exposure_usd=290.0)
    s = _settings()
    cap = buy_headroom_usd(port, "m1", 0.5, s)
    assert cap == 10.0  # min(100, 300-290, 125)


def test_buy_cap_shrinks_with_market_exposure() -> None:
    port = PortfolioSnapshot(
        cash=1000.0,
        positions={"m1": MarketPosition(market_id="m1", yes_shares=50.0, no_shares=0.0)},
        exposure_usd=25.0,
    )
    s = _settings()
    mkt_exp = market_exposure_usd(50.0, 0.0, 0.5)
    assert mkt_exp == 25.0
    cap = buy_headroom_usd(port, "m1", 0.5, s)
    assert cap == 75.0  # min(100-25, 275, 250)


def test_build_risk_execution_state_fields() -> None:
    out = build_risk_execution_state(
        _state(yes_sh=10.0, yes_mid=0.4, exposure_usd=4.0),
        _settings(),
    )
    assert out["pos"] == "LONG_YES"
    assert out["yes_sh"] == 10.0
    assert out["no_sh"] == 0.0
    assert out["mkt_exp"] == 4.0
    assert out["eq"] == 1004.0
    assert out["sell_yes_cap"] == 4.0
    assert out["sell_no_cap"] == 0.0
    assert out["buy_cap"] == 96.0
    assert out["exp_pct"] == round(4.0 / 300.0, 4)
    assert out["tte"] == 120.0
    assert out["edge"] == compute_edge(0.6, 0.5)


def test_risk_payload_json_serializable() -> None:
    raw = risk_payload(_state(), _settings())
    parsed = json.loads(raw)
    for key in (
        "pos",
        "yes_sh",
        "no_sh",
        "mkt_exp",
        "eq",
        "buy_cap",
        "sell_yes_cap",
        "sell_no_cap",
        "exp_pct",
        "cash",
    ):
        assert key in parsed
