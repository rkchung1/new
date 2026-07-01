"""LangGraph workflow: analysts, risk policy, deterministic execution.

Mock mode runs price → poly sequentially so mock_poly_view receives price_view.
LLM mode runs price and poly in parallel, then risk.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from openai import LengthFinishReasonError
from pydantic import ValidationError

from btc5m_agents.agents import prompts
from btc5m_agents.agents.execution import risk_payload, risk_view_to_decision
from btc5m_agents.agents.llm import build_chat_model, with_structured_schema
from btc5m_agents.agents.mock_llm import (
    mock_poly_view,
    mock_price_view,
    mock_risk_view,
)
from btc5m_agents.agents.pricing import poly_view_from_llm
from btc5m_agents.agents.schemas import PolyView, PolyViewLLM, PriceView, RiskView
from btc5m_agents.agents.state import GraphState
from btc5m_agents.config import Settings, get_settings
from btc5m_agents.features.models import MarketDynamicsFeatures, PredictionMarketFeatures
from btc5m_agents.types import PortfolioSnapshot


def _dynamics_payload(state: GraphState) -> str:
    return json.dumps(state.get("dynamics_features"), default=str, separators=(",", ":"))


def _prediction_market_payload(state: GraphState) -> str:
    return json.dumps(
        state.get("prediction_market_features"),
        default=str,
        separators=(",", ":"),
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
        dynamics = MarketDynamicsFeatures.model_validate(state["dynamics_features"])
        port = PortfolioSnapshot.model_validate(state["portfolio"])
        if use_mock:
            pv = mock_price_view(dynamics, port)
            return {"price_view": pv.model_dump()}
        structured = with_structured_schema(llm, PriceView, s, llm_backend=llm_backend)  # type: ignore[arg-type]
        msg = HumanMessage(content=_dynamics_payload(state))
        try:
            pv = structured.invoke([SystemMessage(content=prompts.PRICE_ANALYST), msg])
        except (LengthFinishReasonError, OutputParserException, ValidationError, ValueError):
            pv = mock_price_view(dynamics, port)
        return {"price_view": pv.model_dump()}

    def polymarket_analyst(state: GraphState) -> dict[str, Any]:
        prediction = PredictionMarketFeatures.model_validate(state["prediction_market_features"])
        port = PortfolioSnapshot.model_validate(state["portfolio"])
        if use_mock:
            pv = PriceView.model_validate(state["price_view"])
            poly = mock_poly_view(prediction, port, price=pv)
            return {"poly_view": poly.model_dump()}
        structured = with_structured_schema(llm, PolyViewLLM, s, llm_backend=llm_backend)  # type: ignore[arg-type]
        msg = HumanMessage(content=_prediction_market_payload(state))
        try:
            raw = structured.invoke([SystemMessage(content=prompts.POLYMARKET_ANALYST), msg])
            poly = poly_view_from_llm(raw, prediction.yes_mid)
        except (LengthFinishReasonError, OutputParserException, ValidationError, ValueError):
            pv = PriceView.model_validate(state["price_view"]) if state.get("price_view") else None
            poly = mock_poly_view(prediction, port, price=pv)
        return {"poly_view": poly.model_dump()}

    def risk_manager(state: GraphState) -> dict[str, Any]:
        dynamics = MarketDynamicsFeatures.model_validate(state["dynamics_features"])
        prediction = PredictionMarketFeatures.model_validate(state["prediction_market_features"])
        port = PortfolioSnapshot.model_validate(state["portfolio"])
        poly = PolyView.model_validate(state["poly_view"])
        if use_mock:
            rv = mock_risk_view(state["market_id"], dynamics, prediction, poly, port, s)
            return {"risk_view": rv.model_dump()}
        structured = with_structured_schema(llm, RiskView, s, llm_backend=llm_backend)  # type: ignore[arg-type]
        msg = HumanMessage(content=risk_payload(state, s))
        try:
            rv = structured.invoke([SystemMessage(content=prompts.RISK_MANAGER), msg])
        except (LengthFinishReasonError, OutputParserException, ValidationError, ValueError):
            rv = mock_risk_view(state["market_id"], dynamics, prediction, poly, port, s)
        return {"risk_view": rv.model_dump()}

    def portfolio_manager(state: GraphState) -> dict[str, Any]:
        risk = RiskView.model_validate(state["risk_view"])
        poly = PolyView.model_validate(state["poly_view"])
        dec = risk_view_to_decision(risk, poly)
        return {"decision": dec.model_dump()}

    g = StateGraph(GraphState)
    g.add_node("price_analyst", price_analyst)
    g.add_node("polymarket_analyst", polymarket_analyst)
    g.add_node("risk_manager", risk_manager)
    g.add_node("portfolio_manager", portfolio_manager)
    if use_mock:
        g.add_edge(START, "price_analyst")
        g.add_edge("price_analyst", "polymarket_analyst")
        g.add_edge("polymarket_analyst", "risk_manager")
    else:
        g.add_edge(START, "price_analyst")
        g.add_edge(START, "polymarket_analyst")
        g.add_edge(["price_analyst", "polymarket_analyst"], "risk_manager")
    g.add_edge("risk_manager", "portfolio_manager")
    g.add_edge("portfolio_manager", END)
    return g.compile()
