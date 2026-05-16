"""Agent workflow state (serializable for decisions.jsonl)."""

from __future__ import annotations

from typing import TypedDict


class GraphState(TypedDict, total=False):
    """LangGraph state: partial updates merged each node."""

    snapshot: dict
    portfolio: dict
    price_view: dict
    poly_view: dict
    risk_view: dict
    decision: dict
