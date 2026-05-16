"""Cash and YES share positions."""

from __future__ import annotations

from dataclasses import dataclass, field

from btc5m_agents.types import Position


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    def position_shares(self, market_id: str) -> float:
        p = self.positions.get(market_id)
        return float(p.shares) if p else 0.0

    def exposure_usd(self, marks: dict[str, float]) -> float:
        total = 0.0
        for mid, pos in self.positions.items():
            if pos.shares <= 0:
                continue
            px = marks.get(mid)
            if px is None:
                continue
            total += pos.shares * px
        return float(total)

    def market_value(self, marks: dict[str, float]) -> float:
        return self.exposure_usd(marks)

    def apply_buy(self, market_id: str, shares: float, price: float) -> None:
        cost = shares * price
        self.cash -= cost
        pos = self.positions.get(market_id) or Position(market_id=market_id, shares=0.0, avg_cost=0.0)
        old_s = pos.shares
        old_c = pos.avg_cost
        new_s = old_s + shares
        if new_s > 0:
            pos.avg_cost = (old_c * old_s + price * shares) / new_s
        pos.shares = new_s
        self.positions[market_id] = pos

    def apply_sell(self, market_id: str, shares: float, price: float) -> float:
        """Returns realized PnL contribution from this sell (vs avg cost)."""
        pos = self.positions.get(market_id)
        if not pos or pos.shares <= 0:
            return 0.0
        sell = min(shares, pos.shares)
        proceeds = sell * price
        pnl = (price - pos.avg_cost) * sell
        self.cash += proceeds
        pos.shares -= sell
        if pos.shares <= 1e-12:
            del self.positions[market_id]
        else:
            self.positions[market_id] = pos
        return float(pnl)

    def settle_yes(self, market_id: str, payout_per_share: float) -> tuple[float, float]:
        """
        YES pays `payout_per_share` (1 or 0) per share. Returns (cash_proceeds, settlement_pnl).
        """
        pos = self.positions.get(market_id)
        if not pos or pos.shares <= 0:
            return 0.0, 0.0
        shares = pos.shares
        proceeds = shares * payout_per_share
        cost_basis = shares * pos.avg_cost
        pnl = proceeds - cost_basis
        self.cash += proceeds
        del self.positions[market_id]
        return float(proceeds), float(pnl)
