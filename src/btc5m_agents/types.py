"""Shared domain types (Pydantic)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Action = Literal["BUY_YES", "SELL_YES", "BUY_NO", "SELL_NO", "HOLD"]
TradeAction = Literal["BUY_YES", "SELL_YES", "BUY_NO", "SELL_NO", "SETTLE_YES", "SETTLE_NO"]
OutcomeSide = Literal["YES", "NO"]


class MarketMeta(BaseModel):
    """One BTC 5m Up/Down market (from Gamma)."""

    event_slug: str
    condition_id: str
    market_id: str
    yes_token_id: str
    no_token_id: Optional[str] = None
    question: Optional[str] = None
    start_ts: int
    end_ts: int
    outcomes: Optional[str] = None
    outcome_prices: Optional[str] = None
    uma_resolution_status: Optional[str] = None
    closed: Optional[bool] = None
    settlement_yes: Optional[float] = None  # 1 if Up/YES won, 0 if Down/NO, None if unresolved


class PriceTick(BaseModel):
    t: int
    p: float


class BtcCandle(BaseModel):
    ts: int
    low: float
    high: float
    open: float
    close: float
    volume: float


class Snapshot(BaseModel):
    """Structured state visible to agents at replay time (no settlement peek)."""

    ts: int
    market_id: str
    yes_token_id: str
    event_slug: str
    yes_price: float
    no_price: float
    mins_to_expiry: float
    btc_price: float
    btc_ret_5m: float
    btc_vol_30m: float
    question: Optional[str] = None
    elapsed_sec: Optional[int] = None
    secs_to_expiry: Optional[float] = None
    bid_yes: Optional[float] = None
    ask_yes: Optional[float] = None
    bid_no: Optional[float] = None
    ask_no: Optional[float] = None
    btc_strike: Optional[float] = None
    btc_gap: Optional[float] = None


class MarketPosition(BaseModel):
    """YES and NO legs for one market."""

    market_id: str
    yes_shares: float = 0.0
    yes_avg_cost: float = 0.0
    no_shares: float = 0.0
    no_avg_cost: float = 0.0


class PortfolioSnapshot(BaseModel):
    cash: float
    positions: dict[str, MarketPosition] = Field(default_factory=dict)
    exposure_usd: float = 0.0


class Trade(BaseModel):
    ts: int
    market_id: str
    action: TradeAction
    shares: float
    price: float
    cost: float
    cash_after: float


class EquityPoint(BaseModel):
    ts: int
    cash: float
    positions_value: float
    total_equity: float
    drawdown_pct: float


class DecisionLogRow(BaseModel):
    ts: int
    market_id: str
    snapshot: Snapshot
    price_view: Optional[dict] = None
    poly_view: Optional[dict] = None
    risk_view: Optional[dict] = None
    decision: Optional[dict] = None
    portfolio_before: PortfolioSnapshot
