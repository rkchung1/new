"""Structured Pydantic outputs for each LangGraph node."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

TradeAction = Literal["BUY_YES", "SELL_YES", "BUY_NO", "SELL_NO", "HOLD"]
ShortSignal = Annotated[str, Field(max_length=24)]


class PriceView(BaseModel):
    direction: Literal["UP", "DOWN", "FLAT"]
    confidence: float = Field(ge=0.0, le=1.0)
    signals: list[ShortSignal] = Field(default_factory=list, max_length=4)


class PolyViewLLM(BaseModel):
    """Structured LLM output — edge is computed in code, not by the model."""

    fair_prob_up: float = Field(ge=0.0, le=1.0)
    signals: list[ShortSignal] = Field(default_factory=list, max_length=4)


class PolyView(BaseModel):
    fair_prob_up: float = Field(ge=0.0, le=1.0)
    edge: float
    signals: list[ShortSignal] = Field(default_factory=list, max_length=4)


class RiskView(BaseModel):
    """Compact execution policy: action + size cap only (no signals — 3B models ramble)."""

    action: TradeAction
    max_size: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)


class Decision(BaseModel):
    action: TradeAction
    size_usd: float = Field(ge=0.0)
    predicted_prob_up: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
