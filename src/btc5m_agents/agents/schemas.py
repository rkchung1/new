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
    implied_prob_no: float = Field(ge=0.0, le=1.0)
    edge_vs_yes: float
    edge_vs_no: float
    rationale: str = ""

    @property
    def edge_vs_price(self) -> float:
        """Alias for YES edge (backward compatible)."""
        return self.edge_vs_yes


class RiskView(BaseModel):
    allow_buy_yes: bool = True
    allow_sell_yes: bool = True
    allow_buy_no: bool = True
    allow_sell_no: bool = True
    max_dollar: float = Field(ge=0.0)
    rationale: str = ""


class Decision(BaseModel):
    action: Literal["BUY_YES", "SELL_YES", "BUY_NO", "SELL_NO", "HOLD"]
    size_usd: float = Field(ge=0.0)
    predicted_prob_up: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
