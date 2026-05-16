"""LangGraph workflow: four agents, sequential for clarity and merge semantics."""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from btc5m_agents.agents import prompts
from btc5m_agents.agents.mock_llm import (
    mock_decision,
    mock_poly_view,
    mock_price_view,
    mock_risk_view,
)
from btc5m_agents.agents.schemas import Decision, PolyView, PriceView, RiskView
from btc5m_agents.agents.state import GraphState
from btc5m_agents.config import Settings, get_settings
from btc5m_agents.types import PortfolioSnapshot, Snapshot


def _payload(state: GraphState) -> str:
    return json.dumps(
        {"snapshot": state.get("snapshot"), "portfolio": state.get("portfolio")},
        default=str,
    )


def build_graph(
    *,
    use_mock: bool,
    model: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> Any:
    """Returns compiled LangGraph (`.invoke` merges partial state updates)."""
    s = settings or get_settings()
    model_name = model or s.openai_model

    def price_analyst(state: GraphState) -> dict[str, Any]:
        snap = Snapshot.model_validate(state["snapshot"])
        port = PortfolioSnapshot.model_validate(state["portfolio"])
        if use_mock:
            pv = mock_price_view(snap, port)
            return {"price_view": pv.model_dump()}
        llm = ChatOpenAI(model=model_name, temperature=0, api_key=s.openai_api_key)
        structured = llm.with_structured_output(PriceView)
        msg = HumanMessage(content=_payload(state))
        pv = structured.invoke([SystemMessage(content=prompts.PRICE_ANALYST), msg])
        return {"price_view": pv.model_dump()}

    def polymarket_analyst(state: GraphState) -> dict[str, Any]:
        snap = Snapshot.model_validate(state["snapshot"])
        port = PortfolioSnapshot.model_validate(state["portfolio"])
        pv = PriceView.model_validate(state["price_view"])
        if use_mock:
            poly = mock_poly_view(snap, pv, port)
            return {"poly_view": poly.model_dump()}
        llm = ChatOpenAI(model=model_name, temperature=0, api_key=s.openai_api_key)
        structured = llm.with_structured_output(PolyView)
        extra = json.dumps({"price_view": state.get("price_view")}, default=str)
        msg = HumanMessage(content=_payload(state) + "\n" + extra)
        poly = structured.invoke([SystemMessage(content=prompts.POLYMARKET_ANALYST), msg])
        return {"poly_view": poly.model_dump()}

    def risk_manager(state: GraphState) -> dict[str, Any]:
        snap = Snapshot.model_validate(state["snapshot"])
        port = PortfolioSnapshot.model_validate(state["portfolio"])
        poly = PolyView.model_validate(state["poly_view"])
        if use_mock:
            rv = mock_risk_view(snap, poly, port, s)
            return {"risk_view": rv.model_dump()}
        llm = ChatOpenAI(model=model_name, temperature=0, api_key=s.openai_api_key)
        structured = llm.with_structured_output(RiskView)
        extra = json.dumps(
            {"price_view": state["price_view"], "poly_view": state["poly_view"]},
            default=str,
        )
        msg = HumanMessage(content=_payload(state) + "\n" + extra)
        rv = structured.invoke([SystemMessage(content=prompts.RISK_MANAGER), msg])
        return {"risk_view": rv.model_dump()}

    def portfolio_manager(state: GraphState) -> dict[str, Any]:
        snap = Snapshot.model_validate(state["snapshot"])
        port = PortfolioSnapshot.model_validate(state["portfolio"])
        poly = PolyView.model_validate(state["poly_view"])
        risk = RiskView.model_validate(state["risk_view"])
        if use_mock:
            dec = mock_decision(snap, poly, risk, port, s)
            return {"decision": dec.model_dump()}
        llm = ChatOpenAI(model=model_name, temperature=0, api_key=s.openai_api_key)
        structured = llm.with_structured_output(Decision)
        extra = json.dumps(
            {
                "price_view": state["price_view"],
                "poly_view": state["poly_view"],
                "risk_view": state["risk_view"],
            },
            default=str,
        )
        msg = HumanMessage(content=_payload(state) + "\n" + extra)
        dec = structured.invoke([SystemMessage(content=prompts.PORTFOLIO_MANAGER), msg])
        return {"decision": dec.model_dump()}

    g = StateGraph(GraphState)
    g.add_node("price_analyst", price_analyst)
    g.add_node("polymarket_analyst", polymarket_analyst)
    g.add_node("risk_manager", risk_manager)
    g.add_node("portfolio_manager", portfolio_manager)
    g.add_edge(START, "price_analyst")
    g.add_edge("price_analyst", "polymarket_analyst")
    g.add_edge("polymarket_analyst", "risk_manager")
    g.add_edge("risk_manager", "portfolio_manager")
    g.add_edge("portfolio_manager", END)
    return g.compile()
