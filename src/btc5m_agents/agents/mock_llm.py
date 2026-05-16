"""Deterministic rule-based stand-ins for all agent schemas (no API calls)."""

from __future__ import annotations

from typing import Optional

from btc5m_agents.agents.schemas import Decision, PolyView, PriceView, RiskView
from btc5m_agents.config import Settings, get_settings
from btc5m_agents.types import PortfolioSnapshot, Snapshot


def mock_price_view(snap: Snapshot, _port: PortfolioSnapshot) -> PriceView:
    r = snap.btc_ret_5m
    if r > 0.0001:
        d = "UP"
        c = min(0.95, 0.5 + abs(r) * 50)
    elif r < -0.0001:
        d = "DOWN"
        c = min(0.95, 0.5 + abs(r) * 50)
    else:
        d = "FLAT"
        c = 0.4
    return PriceView(direction=d, confidence=float(c), rationale=f"btc_ret_5m={r:.5f}")


def mock_poly_view(snap: Snapshot, price: PriceView, _port: PortfolioSnapshot) -> PolyView:
    implied = float(max(0.01, min(0.99, snap.yes_price)))
    if price.direction == "UP":
        fair_up = float(price.confidence)
    elif price.direction == "DOWN":
        fair_up = float(1.0 - price.confidence)
    else:
        fair_up = implied
    fair_up = float(max(0.01, min(0.99, fair_up)))
    edge = fair_up - implied
    return PolyView(
        implied_prob_up=implied,
        edge_vs_price=edge,
        rationale=f"dir={price.direction} fair_up={fair_up:.3f} mkt={implied:.3f}",
    )


def mock_risk_view(
    snap: Snapshot,
    poly: PolyView,
    port: PortfolioSnapshot,
    settings: Optional[Settings] = None,
) -> RiskView:
    s = settings or get_settings()
    allow_buy = port.cash > 1.0 and snap.mins_to_expiry > 0.5
    allow_sell = snap.market_id in port.positions and port.positions[snap.market_id].shares > 0
    max_d = min(
        s.max_position_per_market_usd,
        max(0.0, s.max_total_exposure_usd - port.exposure_usd),
        max(0.0, port.cash * 0.25),
    )
    if poly.edge_vs_price < 0.02:
        allow_buy = False
    return RiskView(
        allow_buy=allow_buy,
        allow_sell=allow_sell,
        max_dollar=float(max_d),
        rationale="mock risk caps",
    )


def mock_decision(
    snap: Snapshot,
    poly: PolyView,
    risk: RiskView,
    port: PortfolioSnapshot,
    settings: Optional[Settings] = None,
) -> Decision:
    s = settings or get_settings()
    pos = port.positions.get(snap.market_id)
    shares = pos.shares if pos else 0.0

    if shares > 0 and poly.edge_vs_price < -0.03 and risk.allow_sell:
        return Decision(
            action="SELL_YES",
            size_usd=min(risk.max_dollar, shares * snap.yes_price),
            predicted_prob_up=poly.implied_prob_up,
            confidence=0.55,
            rationale="exit negative edge",
        )
    if poly.edge_vs_price > 0.05 and risk.allow_buy and risk.max_dollar > 1.0:
        return Decision(
            action="BUY_YES",
            size_usd=min(risk.max_dollar, s.max_position_per_market_usd),
            predicted_prob_up=float(max(0.01, min(0.99, poly.implied_prob_up + poly.edge_vs_price))),
            confidence=min(0.9, 0.5 + abs(poly.edge_vs_price) * 3),
            rationale="positive edge vs YES",
        )
    return Decision(
        action="HOLD",
        size_usd=0.0,
        predicted_prob_up=poly.implied_prob_up,
        confidence=0.35,
        rationale="no trade",
    )
