"""Compact execution-state builders and risk-to-decision mapping."""

from __future__ import annotations

import json
from typing import Any, Literal, Optional

from btc5m_agents.agents.schemas import Decision, PolyView, RiskView
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
    btc = state.get("btc_features") or {}

    cap = s.max_total_exposure_usd
    exp_pct = round(port.exposure_usd / cap, 4) if cap > 0 else 0.0

    out: dict[str, Any] = {
        "pos": _position_side(port, mid),
        "exp_pct": exp_pct,
        "cash": round(port.cash, 2),
        "dir": pv.get("direction"),
        "conf": round(float(pv.get("confidence", 0.0)), 3),
        "fair": round(float(poly.get("fair_prob_up", 0.5)), 4),
        "edge": round(float(poly.get("edge", 0.0)), 4),
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
