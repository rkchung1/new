"""Conservative simulated execution with slippage and exposure caps."""

from __future__ import annotations

from typing import Optional

from btc5m_agents.agents.schemas import Decision
from btc5m_agents.config import Settings, get_settings
from btc5m_agents.types import Trade


def _clip_exec_price(yes_price: float, side: str, s: Settings) -> float:
    if side == "buy":
        p = yes_price + s.slippage
    else:
        p = yes_price - s.slippage
    return float(max(s.price_floor, min(s.price_ceiling, p)))


def maybe_execute(
    decision: Decision,
    *,
    ts: int,
    market_id: str,
    yes_price: float,
    cash: float,
    position_shares: float,
    current_mark_exposure_usd: float,
    total_exposure_usd: float,
    settings: Optional[Settings] = None,
) -> Optional[Trade]:
    """
    Returns a Trade if executed, else None (effective HOLD).
    BUY charges exec price; SELL receives exec price.
    """
    s = settings or get_settings()
    if decision.action == "HOLD" or decision.size_usd <= 0:
        return None

    if decision.action == "BUY_YES":
        if not yes_price or yes_price != yes_price:  # NaN guard
            return None
        px = _clip_exec_price(yes_price, "buy", s)
        max_add_usd = max(0.0, s.max_position_per_market_usd - current_mark_exposure_usd)
        max_add_usd = min(max_add_usd, decision.size_usd, s.max_position_per_market_usd)
        max_total = max(0.0, s.max_total_exposure_usd - total_exposure_usd)
        spend = min(max_add_usd, max_total, cash * 0.999, decision.size_usd)
        if spend < 1.0:
            return None
        shares = spend / px
        if shares <= 0:
            return None
        cost = shares * px
        return Trade(
            ts=ts,
            market_id=market_id,
            action="BUY_YES",
            shares=float(shares),
            price=float(px),
            cost=float(cost),
            cash_after=float(cash - cost),
        )

    if decision.action == "SELL_YES":
        if position_shares <= 0:
            return None
        px = _clip_exec_price(yes_price, "sell", s)
        max_usd = min(decision.size_usd, position_shares * px)
        shares = min(position_shares, max_usd / px if px > 0 else 0.0)
        if shares <= 0:
            return None
        proceeds = shares * px
        return Trade(
            ts=ts,
            market_id=market_id,
            action="SELL_YES",
            shares=float(shares),
            price=float(px),
            cost=float(-proceeds),
            cash_after=float(cash + proceeds),
        )

    return None
