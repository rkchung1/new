"""Deterministic rule-based stand-ins for all agent schemas (no API calls)."""

from __future__ import annotations

from typing import Optional

from btc5m_agents.agents.schemas import Decision, PolyView, PriceView, RiskView
from btc5m_agents.config import Settings, get_settings
from btc5m_agents.features.models import BtcFeatures, PolyFeatures
from btc5m_agents.types import PortfolioSnapshot


def _gap_signal(btc: BtcFeatures) -> tuple[str, float, str]:
    if btc.strike_gap_pct is not None and btc.strike_gap_pct == btc.strike_gap_pct:
        rel = btc.strike_gap_pct
        if rel > 0.0001:
            return "UP", min(0.95, 0.5 + abs(rel) * 500), f"strike_gap_pct={rel:.6f}"
        if rel < -0.0001:
            return "DOWN", min(0.95, 0.5 + abs(rel) * 500), f"strike_gap_pct={rel:.6f}"
        return "FLAT", 0.4, f"strike_gap_pct={rel:.6f}"
    r = btc.return_60s
    if r is not None and r == r:
        if r > 0.0001:
            return "UP", min(0.95, 0.5 + abs(r) * 50), f"return_60s={r:.5f}"
        if r < -0.0001:
            return "DOWN", min(0.95, 0.5 + abs(r) * 50), f"return_60s={r:.5f}"
        return "FLAT", 0.4, f"return_60s={r:.5f}"
    return "FLAT", 0.4, "no momentum signal"


def _fair_probs(
    poly: PolyFeatures,
    price: Optional[PriceView] = None,
) -> tuple[float, float]:
    implied_up = float(max(0.01, min(0.99, poly.yes_mid)))
    implied_no = float(max(0.01, min(0.99, poly.no_mid)))
    if price is not None:
        if price.direction == "UP":
            fair_up = float(price.confidence)
        elif price.direction == "DOWN":
            fair_up = float(1.0 - price.confidence)
        else:
            fair_up = implied_up
    else:
        fair_up = implied_up
    fair_up = float(max(0.01, min(0.99, fair_up)))
    fair_no = float(max(0.01, min(0.99, 1.0 - fair_up)))
    return fair_up, fair_no


def mock_price_view(btc: BtcFeatures, _port: PortfolioSnapshot) -> PriceView:
    d, c, rationale = _gap_signal(btc)
    return PriceView(direction=d, confidence=float(c), rationale=rationale)


def mock_poly_view(
    poly: PolyFeatures,
    _port: PortfolioSnapshot,
    price: Optional[PriceView] = None,
) -> PolyView:
    implied_up = float(max(0.01, min(0.99, poly.yes_mid)))
    implied_no = float(max(0.01, min(0.99, poly.no_mid)))
    fair_up, fair_no = _fair_probs(poly, price)
    edge_yes = fair_up - implied_up
    edge_no = fair_no - implied_no
    d = price.direction if price else "FLAT"
    return PolyView(
        implied_prob_up=implied_up,
        implied_prob_no=implied_no,
        edge_vs_yes=edge_yes,
        edge_vs_no=edge_no,
        rationale=f"dir={d} fair_up={fair_up:.3f} mkt_yes={implied_up:.3f} fair_no={fair_no:.3f} mkt_no={implied_no:.3f}",
    )


def mock_risk_view(
    btc: BtcFeatures,
    poly: PolyView,
    port: PortfolioSnapshot,
    settings: Optional[Settings] = None,
) -> RiskView:
    s = settings or get_settings()
    secs_left = btc.secs_to_expiry if btc.secs_to_expiry is not None else 300.0
    can_trade = port.cash > 1.0 and secs_left > 30.0
    pos = port.positions.get(btc.market_id)
    yes_shares = pos.yes_shares if pos else 0.0
    no_shares = pos.no_shares if pos else 0.0
    max_d = min(
        s.max_position_per_market_usd,
        max(0.0, s.max_total_exposure_usd - port.exposure_usd),
        max(0.0, port.cash * 0.25),
    )
    return RiskView(
        allow_buy_yes=can_trade and poly.edge_vs_yes > 0.02,
        allow_sell_yes=yes_shares > 0,
        allow_buy_no=can_trade and poly.edge_vs_no > 0.02,
        allow_sell_no=no_shares > 0,
        max_dollar=float(max_d),
        rationale="mock risk caps",
    )


def mock_decision(
    poly_f: PolyFeatures,
    poly: PolyView,
    risk: RiskView,
    port: PortfolioSnapshot,
    settings: Optional[Settings] = None,
) -> Decision:
    s = settings or get_settings()
    pos = port.positions.get(poly_f.market_id)
    yes_shares = pos.yes_shares if pos else 0.0
    no_shares = pos.no_shares if pos else 0.0
    yes_mid = poly_f.yes_mid
    no_mid = poly_f.no_mid

    if yes_shares > 0 and poly.edge_vs_yes < -0.03 and risk.allow_sell_yes:
        return Decision(
            action="SELL_YES",
            size_usd=min(risk.max_dollar, yes_shares * yes_mid),
            predicted_prob_up=poly.implied_prob_up,
            confidence=0.55,
            rationale="exit negative YES edge",
        )
    if no_shares > 0 and poly.edge_vs_no < -0.03 and risk.allow_sell_no:
        return Decision(
            action="SELL_NO",
            size_usd=min(risk.max_dollar, no_shares * no_mid),
            predicted_prob_up=poly.implied_prob_up,
            confidence=0.55,
            rationale="exit negative NO edge",
        )
    if poly.edge_vs_yes > 0.05 and risk.allow_buy_yes and risk.max_dollar > 1.0:
        return Decision(
            action="BUY_YES",
            size_usd=min(risk.max_dollar, s.max_position_per_market_usd),
            predicted_prob_up=float(max(0.01, min(0.99, poly.implied_prob_up + poly.edge_vs_yes))),
            confidence=min(0.9, 0.5 + abs(poly.edge_vs_yes) * 3),
            rationale="positive edge vs YES",
        )
    if poly.edge_vs_no > 0.05 and risk.allow_buy_no and risk.max_dollar > 1.0:
        return Decision(
            action="BUY_NO",
            size_usd=min(risk.max_dollar, s.max_position_per_market_usd),
            predicted_prob_up=float(max(0.01, min(0.99, 1.0 - (poly.implied_prob_no + poly.edge_vs_no)))),
            confidence=min(0.9, 0.5 + abs(poly.edge_vs_no) * 3),
            rationale="positive edge vs NO",
        )
    return Decision(
        action="HOLD",
        size_usd=0.0,
        predicted_prob_up=poly.implied_prob_up,
        confidence=0.35,
        rationale="no trade",
    )
