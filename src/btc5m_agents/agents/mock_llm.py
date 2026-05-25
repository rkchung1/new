"""Deterministic rule-based stand-ins for all agent schemas (no API calls)."""

from __future__ import annotations

from typing import Optional

from btc5m_agents.agents.schemas import PolyView, PriceView, RiskView
from btc5m_agents.config import Settings, get_settings
from btc5m_agents.features.models import BtcFeatures, PolyFeatures
from btc5m_agents.types import PortfolioSnapshot


def _direction_signal(btc: BtcFeatures) -> tuple[str, float, list[str]]:
    signals: list[str] = []
    if btc.strike_gap_pct is not None and btc.strike_gap_pct == btc.strike_gap_pct:
        rel = btc.strike_gap_pct
        if rel > 0.0001:
            signals.append("above_strike")
            return "UP", min(0.95, 0.5 + abs(rel) * 500), signals
        if rel < -0.0001:
            signals.append("below_strike")
            return "DOWN", min(0.95, 0.5 + abs(rel) * 500), signals
        signals.append("at_strike")
        return "FLAT", 0.4, signals
    r = btc.return_60s
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


def _fair_prob_up(poly: PolyFeatures, price: Optional[PriceView] = None) -> float:
    mkt = float(max(0.01, min(0.99, poly.yes_mid)))
    if price is not None:
        if price.direction == "UP":
            fair = float(price.confidence)
        elif price.direction == "DOWN":
            fair = float(1.0 - price.confidence)
        else:
            fair = mkt
    else:
        fair = mkt
    return float(max(0.01, min(0.99, fair)))


def mock_price_view(btc: BtcFeatures, _port: PortfolioSnapshot) -> PriceView:
    d, c, signals = _direction_signal(btc)
    if btc.gap_change_30s is not None and btc.gap_change_30s == btc.gap_change_30s:
        if btc.gap_change_30s > 0:
            signals.append("gap_widening")
        elif btc.gap_change_30s < 0:
            signals.append("gap_narrowing")
    return PriceView(direction=d, confidence=float(c), signals=signals[:4])


def mock_poly_view(
    poly: PolyFeatures,
    _port: PortfolioSnapshot,
    price: Optional[PriceView] = None,
) -> PolyView:
    mkt = float(max(0.01, min(0.99, poly.yes_mid)))
    fair = _fair_prob_up(poly, price)
    edge = fair - mkt
    signals: list[str] = []
    if poly.prob_divergence and abs(poly.prob_divergence) > 0.01:
        signals.append("mispriced_sum")
    if edge > 0.02:
        signals.append("yes_cheap")
    elif edge < -0.02:
        signals.append("no_cheap")
    return PolyView(fair_prob_up=fair, edge=edge, signals=signals[:4])


def _cap_size(s: Settings, port: PortfolioSnapshot) -> float:
    return float(
        min(
            s.max_position_per_market_usd,
            max(0.0, s.max_total_exposure_usd - port.exposure_usd),
            max(0.0, port.cash * 0.25),
        )
    )


def mock_risk_view(
    market_id: str,
    btc: BtcFeatures,
    poly: PolyView,
    poly_f: PolyFeatures,
    port: PortfolioSnapshot,
    settings: Optional[Settings] = None,
) -> RiskView:
    s = settings or get_settings()
    secs_left = btc.secs_to_expiry if btc.secs_to_expiry is not None else 300.0
    can_trade = port.cash > 1.0 and secs_left > 30.0
    pos = port.positions.get(market_id)
    yes_shares = pos.yes_shares if pos else 0.0
    no_shares = pos.no_shares if pos else 0.0
    mkt_yes = float(max(0.01, min(0.99, poly_f.yes_mid)))
    no_mid = float(max(0.01, min(0.99, 1.0 - mkt_yes)))

    max_d = _cap_size(s, port)

    if secs_left <= 30.0:
        return RiskView(action="HOLD", max_size=0.0, confidence=0.35)

    if yes_shares > 0 and poly.edge < -0.03:
        return RiskView(
            action="SELL_YES",
            max_size=min(max_d, yes_shares * mkt_yes),
            confidence=0.55,
        )
    if no_shares > 0 and poly.edge > 0.03:
        return RiskView(
            action="SELL_NO",
            max_size=min(max_d, no_shares * no_mid),
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
