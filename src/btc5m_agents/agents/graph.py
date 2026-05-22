"""LangGraph workflow: parallel analysts, then risk and portfolio managers."""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from btc5m_agents.agents import prompts
from btc5m_agents.agents.llm import build_chat_model
from btc5m_agents.agents.mock_llm import (
    mock_decision,
    mock_poly_view,
    mock_price_view,
    mock_risk_view,
)
from btc5m_agents.agents.schemas import Decision, PolyView, PriceView, RiskView
from btc5m_agents.agents.state import GraphState
from btc5m_agents.config import Settings, get_settings
from btc5m_agents.features.models import BtcFeatures, PolyFeatures
from btc5m_agents.types import PortfolioSnapshot


def _price_payload(state: GraphState) -> str:
    return json.dumps(
        {
            "market_id": state.get("market_id"),
            "ts": state.get("ts"),
            "btc_features": state.get("btc_features"),
        },
        default=str,
    )


def _poly_payload(state: GraphState) -> str:
    return json.dumps(
        {
            "market_id": state.get("market_id"),
            "ts": state.get("ts"),
            "poly_features": state.get("poly_features"),
        },
        default=str,
    )


def _risk_payload(state: GraphState) -> str:
    btc = state.get("btc_features") or {}
    return json.dumps(
        {
            "market_id": state.get("market_id"),
            "ts": state.get("ts"),
            "secs_to_expiry": btc.get("secs_to_expiry"),
            "portfolio": state.get("portfolio"),
            "price_view": state.get("price_view"),
            "poly_view": state.get("poly_view"),
        },
        default=str,
    )


def _portfolio_payload(state: GraphState) -> str:
    return json.dumps(
        {
            "market_id": state.get("market_id"),
            "ts": state.get("ts"),
            "portfolio": state.get("portfolio"),
            "price_view": state.get("price_view"),
            "poly_view": state.get("poly_view"),
            "risk_view": state.get("risk_view"),
        },
        default=str,
    )


def build_graph(
    *,
    use_mock: bool,
    model: Optional[str] = None,
    settings: Optional[Settings] = None,
    llm_backend: Optional[str] = None,
    vllm_base_url: Optional[str] = None,
) -> Any:
    """Returns compiled LangGraph (`.invoke` merges partial state updates)."""
    s = settings or get_settings()
    llm = None
    if not use_mock:
        llm = build_chat_model(
            s,
            model=model,
            llm_backend=llm_backend,
            vllm_base_url=vllm_base_url,
        )

    def price_analyst(state: GraphState) -> dict[str, Any]:
        btc = BtcFeatures.model_validate(state["btc_features"])
        port = PortfolioSnapshot.model_validate(state["portfolio"])
        if use_mock:
            pv = mock_price_view(btc, port)
            return {"price_view": pv.model_dump()}
        structured = llm.with_structured_output(PriceView)  # type: ignore[union-attr]
        msg = HumanMessage(content=_price_payload(state))
        pv = structured.invoke([SystemMessage(content=prompts.PRICE_ANALYST), msg])
        return {"price_view": pv.model_dump()}

    def polymarket_analyst(state: GraphState) -> dict[str, Any]:
        poly_f = PolyFeatures.model_validate(state["poly_features"])
        port = PortfolioSnapshot.model_validate(state["portfolio"])
        if use_mock:
            poly = mock_poly_view(poly_f, port)
            return {"poly_view": poly.model_dump()}
        structured = llm.with_structured_output(PolyView)  # type: ignore[union-attr]
        msg = HumanMessage(content=_poly_payload(state))
        poly = structured.invoke([SystemMessage(content=prompts.POLYMARKET_ANALYST), msg])
        return {"poly_view": poly.model_dump()}

    def risk_manager(state: GraphState) -> dict[str, Any]:
        btc = BtcFeatures.model_validate(state["btc_features"])
        port = PortfolioSnapshot.model_validate(state["portfolio"])
        poly = PolyView.model_validate(state["poly_view"])
        if use_mock:
            rv = mock_risk_view(btc, poly, port, s)
            return {"risk_view": rv.model_dump()}
        structured = llm.with_structured_output(RiskView)  # type: ignore[union-attr]
        msg = HumanMessage(content=_risk_payload(state))
        rv = structured.invoke([SystemMessage(content=prompts.RISK_MANAGER), msg])
        return {"risk_view": rv.model_dump()}

    def portfolio_manager(state: GraphState) -> dict[str, Any]:
        poly_f = PolyFeatures.model_validate(state["poly_features"])
        port = PortfolioSnapshot.model_validate(state["portfolio"])
        poly = PolyView.model_validate(state["poly_view"])
        risk = RiskView.model_validate(state["risk_view"])
        if use_mock:
            dec = mock_decision(poly_f, poly, risk, port, s)
            return {"decision": dec.model_dump()}
        structured = llm.with_structured_output(Decision)  # type: ignore[union-attr]
        msg = HumanMessage(content=_portfolio_payload(state))
        dec = structured.invoke([SystemMessage(content=prompts.PORTFOLIO_MANAGER), msg])
        return {"decision": dec.model_dump()}

    g = StateGraph(GraphState)
    g.add_node("price_analyst", price_analyst)
    g.add_node("polymarket_analyst", polymarket_analyst)
    g.add_node("risk_manager", risk_manager)
    g.add_node("portfolio_manager", portfolio_manager)
    g.add_edge(START, "price_analyst")
    g.add_edge(START, "polymarket_analyst")
    g.add_edge(["price_analyst", "polymarket_analyst"], "risk_manager")
    g.add_edge("risk_manager", "portfolio_manager")
    g.add_edge("portfolio_manager", END)
    return g.compile()
