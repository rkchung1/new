"""Deterministic rule-based stand-ins for all agent schemas (no API calls)."""

from __future__ import annotations

from typing import Optional

from btc5m_agents.agents.pricing import build_poly_view
from btc5m_agents.agents.schemas import PolyView, PriceView, RiskView
from btc5m_agents.agents.sizing import buy_headroom_usd, position_legs, sell_caps
from btc5m_agents.config import Settings, get_settings
from btc5m_agents.features.models import MarketDynamicsFeatures, PredictionMarketFeatures
from btc5m_agents.types import PortfolioSnapshot


def _direction_signal(dynamics: MarketDynamicsFeatures) -> tuple[str, float, list[str]]:
    signals: list[str] = []
    if dynamics.strike_gap_pct is not None and dynamics.strike_gap_pct == dynamics.strike_gap_pct:
        rel = dynamics.strike_gap_pct
        if rel > 0.0001:
            signals.append("above_strike")
            return "UP", min(0.95, 0.5 + abs(rel) * 500), signals
        if rel < -0.0001:
            signals.append("below_strike")
            return "DOWN", min(0.95, 0.5 + abs(rel) * 500), signals
        signals.append("at_strike")
        return "FLAT", 0.4, signals
    r = dynamics.return_60s
    if r is not None and r == r:
        if r > 0.0001:
            signals.append("mom_up")
            return "UP", min(0.95, 0.5 + abs(r) * 50), signals
        if r < -0.0001:
            signals.append("mom_down")
            return "DOWN", min(0.95, 0.5 + abs(r) * 50), signals
        signals.append("mom_flat")
        return "FLAT", 0.4, signals
    return "FLAT", 0.4, ["weak_signal"]


def _fair_prob_up(
    prediction: PredictionMarketFeatures,
    price: Optional[PriceView] = None,
) -> float:
    mkt = float(max(0.01, min(0.99, prediction.yes_mid)))
    if prediction.model_prob_up is not None and prediction.model_prob_up == prediction.model_prob_up:
        fair = float(prediction.model_prob_up)
    elif price is not None:
        if price.direction == "UP":
            fair = float(price.confidence)
        elif price.direction == "DOWN":
            fair = float(1.0 - price.confidence)
        else:
            fair = mkt
    else:
        fair = mkt
    return float(max(0.01, min(0.99, fair)))


def mock_price_view(dynamics: MarketDynamicsFeatures, _port: PortfolioSnapshot) -> PriceView:
    d, c, signals = _direction_signal(dynamics)
    if dynamics.gap_change_30s is not None and dynamics.gap_change_30s == dynamics.gap_change_30s:
        if dynamics.gap_change_30s > 0:
            signals.append("gap_widening")
        elif dynamics.gap_change_30s < 0:
            signals.append("gap_narrowing")
    return PriceView(direction=d, confidence=float(c), signals=signals[:4])


def mock_poly_view(
    prediction: PredictionMarketFeatures,
    _port: PortfolioSnapshot,
    price: Optional[PriceView] = None,
) -> PolyView:
    fair = _fair_prob_up(prediction, price)
    signals: list[str] = []
    if prediction.prob_divergence and abs(prediction.prob_divergence) > 0.01:
        signals.append("mispriced_sum")
    edge_preview = fair - float(prediction.yes_mid)
    if edge_preview > 0.02:
        signals.append("yes_cheap")
    elif edge_preview < -0.02:
        signals.append("no_cheap")
    if prediction.model_vs_market is not None and abs(prediction.model_vs_market) > 0.03:
        signals.append("model_diverge")
    return build_poly_view(fair, signals, prediction.yes_mid)


def mock_risk_view(
    market_id: str,
    dynamics: MarketDynamicsFeatures,
    prediction: PredictionMarketFeatures,
    poly: PolyView,
    port: PortfolioSnapshot,
    settings: Optional[Settings] = None,
) -> RiskView:
    s = settings or get_settings()
    secs_left = dynamics.secs_to_expiry if dynamics.secs_to_expiry is not None else 300.0
    can_trade = port.cash > 1.0 and secs_left > 30.0
    yes_shares, no_shares = position_legs(port, market_id)
    mkt_yes = float(max(0.01, min(0.99, prediction.yes_mid)))
    sell_yes_cap, sell_no_cap = sell_caps(yes_shares, no_shares, mkt_yes)

    max_d = buy_headroom_usd(port, market_id, mkt_yes, s)

    if secs_left <= 30.0:
        return RiskView(action="HOLD", max_size=0.0, confidence=0.35)

    if yes_shares > 0 and poly.edge < -0.03:
        return RiskView(
            action="SELL_YES",
            max_size=min(max_d, sell_yes_cap),
            confidence=0.55,
        )
    if no_shares > 0 and poly.edge > 0.03:
        return RiskView(
            action="SELL_NO",
            max_size=min(max_d, sell_no_cap),
            confidence=0.55,
        )
    if poly.edge > 0.05 and can_trade and max_d > 1.0:
        return RiskView(
            action="BUY_YES",
            max_size=min(max_d, s.max_position_per_market_usd),
            confidence=min(0.9, 0.5 + abs(poly.edge) * 3),
        )
    if poly.edge < -0.05 and can_trade and max_d > 1.0:
        return RiskView(
            action="BUY_NO",
            max_size=min(max_d, s.max_position_per_market_usd),
            confidence=min(0.9, 0.5 + abs(poly.edge) * 3),
        )

    return RiskView(action="HOLD", max_size=0.0, confidence=0.35)
