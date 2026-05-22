"""Conservative simulated execution with slippage and exposure caps."""

from __future__ import annotations

from typing import Optional

from btc5m_agents.agents.schemas import Decision
from btc5m_agents.config import Settings, get_settings
from btc5m_agents.types import Trade


def _clip_exec_price(price: float, side: str, s: Settings) -> float:
    if side == "buy":
        p = price + s.slippage
    else:
        p = price - s.slippage
    return float(max(s.price_floor, min(s.price_ceiling, p)))


def maybe_execute(
    decision: Decision,
    *,
    ts: int,
    market_id: str,
    yes_price: float,
    no_price: float,
    bid_yes: Optional[float] = None,
    ask_yes: Optional[float] = None,
    bid_no: Optional[float] = None,
    ask_no: Optional[float] = None,
    cash: float,
    yes_shares: float,
    no_shares: float,
    current_mark_exposure_usd: float,
    total_exposure_usd: float,
    settings: Optional[Settings] = None,
) -> Optional[Trade]:
    """
    Returns a Trade if executed, else None (effective HOLD).
    BUY pays ask (+ slippage); SELL receives bid (- slippage).
    """
    s = settings or get_settings()
    if decision.action == "HOLD" or decision.size_usd <= 0:
        return None

    if decision.action == "BUY_YES":
        ref = ask_yes if ask_yes is not None and ask_yes == ask_yes else yes_price
        if not ref or ref != ref:
            return None
        px = _clip_exec_price(ref, "buy", s)
        return _execute_buy(
            ts, market_id, "BUY_YES", px, cash, current_mark_exposure_usd, total_exposure_usd, decision, s
        )

    if decision.action == "SELL_YES":
        if yes_shares <= 0:
            return None
        ref = bid_yes if bid_yes is not None and bid_yes == bid_yes else yes_price
        px = _clip_exec_price(ref, "sell", s)
        return _execute_sell(ts, market_id, "SELL_YES", px, yes_shares, cash, decision)

    if decision.action == "BUY_NO":
        ref = ask_no if ask_no is not None and ask_no == ask_no else no_price
        if not ref or ref != ref:
            return None
        px = _clip_exec_price(ref, "buy", s)
        return _execute_buy(
            ts, market_id, "BUY_NO", px, cash, current_mark_exposure_usd, total_exposure_usd, decision, s
        )

    if decision.action == "SELL_NO":
        if no_shares <= 0:
            return None
        ref = bid_no if bid_no is not None and bid_no == bid_no else no_price
        px = _clip_exec_price(ref, "sell", s)
        return _execute_sell(ts, market_id, "SELL_NO", px, no_shares, cash, decision)

    return None


def _execute_buy(
    ts: int,
    market_id: str,
    action: str,
    px: float,
    cash: float,
    current_mark_exposure_usd: float,
    total_exposure_usd: float,
    decision: Decision,
    s: Settings,
) -> Optional[Trade]:
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
        action=action,  # type: ignore[arg-type]
        shares=float(shares),
        price=float(px),
        cost=float(cost),
        cash_after=float(cash - cost),
    )


def _execute_sell(
    ts: int,
    market_id: str,
    action: str,
    px: float,
    position_shares: float,
    cash: float,
    decision: Decision,
) -> Optional[Trade]:
    max_usd = min(decision.size_usd, position_shares * px)
    shares = min(position_shares, max_usd / px if px > 0 else 0.0)
    if shares <= 0:
        return None
    proceeds = shares * px
    return Trade(
        ts=ts,
        market_id=market_id,
        action=action,  # type: ignore[arg-type]
        shares=float(shares),
        price=float(px),
        cost=float(-proceeds),
        cash_after=float(cash + proceeds),
    )
