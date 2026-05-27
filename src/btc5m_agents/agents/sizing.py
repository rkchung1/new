"""Shared portfolio sizing helpers for risk state and mock agents."""

from __future__ import annotations

from typing import Optional, Tuple

from btc5m_agents.config import Settings, get_settings
from btc5m_agents.types import PortfolioSnapshot


def clip_yes_mid(yes_mid: float) -> float:
    return float(max(0.01, min(0.99, yes_mid)))


def market_exposure_usd(yes_sh: float, no_sh: float, yes_mid: float) -> float:
    """Marked notional for one market (YES + NO legs)."""
    mid = clip_yes_mid(yes_mid)
    no_mid = 1.0 - mid
    return float(yes_sh * mid + no_sh * no_mid)


def position_legs(
    port: PortfolioSnapshot,
    market_id: str,
) -> Tuple[float, float]:
    pos = port.positions.get(market_id)
    if not pos:
        return 0.0, 0.0
    return float(pos.yes_shares), float(pos.no_shares)


def sell_caps(yes_sh: float, no_sh: float, yes_mid: float) -> Tuple[float, float]:
    """Max SELL_YES / SELL_NO notional at current mids."""
    mid = clip_yes_mid(yes_mid)
    no_mid = 1.0 - mid
    return float(yes_sh * mid), float(no_sh * no_mid)


def buy_headroom_usd(
    port: PortfolioSnapshot,
    market_id: str,
    yes_mid: float,
    settings: Optional[Settings] = None,
) -> float:
    """
    Max new buy notional this step for the current market.
    Mirrors mock_risk_view _cap_size with per-market headroom.
    """
    s = settings or get_settings()
    yes_sh, no_sh = position_legs(port, market_id)
    mkt_exp = market_exposure_usd(yes_sh, no_sh, yes_mid)
    return float(
        min(
            max(0.0, s.max_position_per_market_usd - mkt_exp),
            max(0.0, s.max_total_exposure_usd - port.exposure_usd),
            max(0.0, port.cash * 0.25),
        )
    )


def portfolio_equity_usd(port: PortfolioSnapshot) -> float:
    """Approx equity: cash + marked exposure (snapshot exposure_usd)."""
    return float(port.cash + port.exposure_usd)
