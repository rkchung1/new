"""Structured Pydantic outputs for each LangGraph node."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PriceView(BaseModel):
    direction: Literal["UP", "DOWN", "FLAT"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class PolyView(BaseModel):
    implied_prob_up: float = Field(ge=0.0, le=1.0)
    edge_vs_price: float
    rationale: str = ""


class RiskView(BaseModel):
    allow_buy: bool = True
    allow_sell: bool = True
    max_dollar: float = Field(ge=0.0)
    rationale: str = ""


class Decision(BaseModel):
    action: Literal["BUY_YES", "SELL_YES", "HOLD"]
    size_usd: float = Field(ge=0.0)
    predicted_prob_up: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
