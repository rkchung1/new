"""Compact execution-state builders and risk-to-decision mapping."""

from __future__ import annotations

import json
from typing import Any, Literal, Optional

from btc5m_agents.agents.pricing import compute_edge
from btc5m_agents.agents.schemas import Decision, PolyView, RiskView
from btc5m_agents.agents.sizing import (
    buy_headroom_usd,
    market_exposure_usd,
    portfolio_equity_usd,
    position_legs,
    sell_caps,
)
from btc5m_agents.agents.state import GraphState
from btc5m_agents.config import Settings, get_settings
from btc5m_agents.types import PortfolioSnapshot

PositionSide = Literal["LONG_YES", "LONG_NO", "NONE"]


def _position_side(port: PortfolioSnapshot, market_id: str) -> PositionSide:
    pos = port.positions.get(market_id)
    if not pos:
        return "NONE"
    if pos.yes_shares > 0:
        return "LONG_YES"
    if pos.no_shares > 0:
        return "LONG_NO"
    return "NONE"


def build_risk_execution_state(
    state: GraphState,
    settings: Optional[Settings] = None,
) -> dict[str, Any]:
    """Tiny synthesized state for the risk policy head (no nested analyst objects)."""
    s = settings or get_settings()
    port = PortfolioSnapshot.model_validate(state["portfolio"])
    mid = str(state["market_id"])
    pv = state.get("price_view") or {}
    poly = state.get("poly_view") or {}
    poly_f = state.get("poly_features") or {}
    btc = state.get("btc_features") or {}

    yes_mid = float(poly_f.get("yes_mid", 0.5))
    yes_sh, no_sh = position_legs(port, mid)
    mkt_exp = market_exposure_usd(yes_sh, no_sh, yes_mid)
    sell_yes_cap, sell_no_cap = sell_caps(yes_sh, no_sh, yes_mid)
    buy_cap = buy_headroom_usd(port, mid, yes_mid, s)

    cap = s.max_total_exposure_usd
    exp_pct = round(port.exposure_usd / cap, 4) if cap > 0 else 0.0

    out: dict[str, Any] = {
        "pos": _position_side(port, mid),
        "yes_sh": round(yes_sh, 4),
        "no_sh": round(no_sh, 4),
        "mkt_exp": round(mkt_exp, 2),
        "eq": round(portfolio_equity_usd(port), 2),
        "buy_cap": round(buy_cap, 2),
        "sell_yes_cap": round(sell_yes_cap, 2),
        "sell_no_cap": round(sell_no_cap, 2),
        "exp_pct": exp_pct,
        "cash": round(port.cash, 2),
        "dir": pv.get("direction"),
        "conf": round(float(pv.get("confidence", 0.0)), 3),
        "fair": round(float(poly.get("fair_prob_up", 0.5)), 4),
        "edge": compute_edge(
            float(poly.get("fair_prob_up", 0.5)),
            yes_mid,
        ),
    }
    tte = btc.get("secs_to_expiry")
    if tte is not None:
        out["tte"] = round(float(tte), 1)
    return out


def risk_payload(state: GraphState, settings: Optional[Settings] = None) -> str:
    return json.dumps(
        build_risk_execution_state(state, settings),
        separators=(",", ":"),
        default=str,
    )


def risk_view_to_decision(risk: RiskView, poly: PolyView) -> Decision:
    """Map compact risk policy output to engine Decision."""
    return Decision(
        action=risk.action,
        size_usd=0.0 if risk.action == "HOLD" else float(risk.max_size),
        predicted_prob_up=float(poly.fair_prob_up),
        confidence=float(risk.confidence),
        rationale="",
    )
