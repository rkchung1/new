"""Cash and YES/NO share positions per market."""

from __future__ import annotations

from dataclasses import dataclass, field

from btc5m_agents.types import MarketPosition, OutcomeSide


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, MarketPosition] = field(default_factory=dict)

    def _pos(self, market_id: str) -> MarketPosition:
        return self.positions.get(market_id) or MarketPosition(market_id=market_id)

    def position_shares(self, market_id: str, side: OutcomeSide) -> float:
        p = self.positions.get(market_id)
        if not p:
            return 0.0
        return float(p.yes_shares if side == "YES" else p.no_shares)

    def exposure_usd(self, yes_marks: dict[str, float], no_marks: dict[str, float]) -> float:
        total = 0.0
        for mid, pos in self.positions.items():
            if pos.yes_shares > 0:
                px = yes_marks.get(mid)
                if px is not None and px == px:
                    total += pos.yes_shares * px
            if pos.no_shares > 0:
                px = no_marks.get(mid)
                if px is not None and px == px:
                    total += pos.no_shares * px
        return float(total)

    def market_exposure_usd(
        self,
        market_id: str,
        yes_marks: dict[str, float],
        no_marks: dict[str, float],
    ) -> float:
        pos = self.positions.get(market_id)
        if not pos:
            return 0.0
        exp = 0.0
        yp = yes_marks.get(market_id)
        if pos.yes_shares > 0 and yp is not None and yp == yp:
            exp += pos.yes_shares * yp
        np_ = no_marks.get(market_id)
        if pos.no_shares > 0 and np_ is not None and np_ == np_:
            exp += pos.no_shares * np_
        return float(exp)

    def market_value(self, yes_marks: dict[str, float], no_marks: dict[str, float]) -> float:
        return self.exposure_usd(yes_marks, no_marks)

    def apply_buy(self, market_id: str, side: OutcomeSide, shares: float, price: float) -> None:
        cost = shares * price
        self.cash -= cost
        pos = self._pos(market_id)
        if side == "YES":
            old_s, old_c = pos.yes_shares, pos.yes_avg_cost
            new_s = old_s + shares
            pos.yes_avg_cost = (old_c * old_s + price * shares) / new_s if new_s > 0 else 0.0
            pos.yes_shares = new_s
        else:
            old_s, old_c = pos.no_shares, pos.no_avg_cost
            new_s = old_s + shares
            pos.no_avg_cost = (old_c * old_s + price * shares) / new_s if new_s > 0 else 0.0
            pos.no_shares = new_s
        self.positions[market_id] = pos

    def apply_sell(self, market_id: str, side: OutcomeSide, shares: float, price: float) -> float:
        pos = self.positions.get(market_id)
        if not pos:
            return 0.0
        if side == "YES":
            held, avg = pos.yes_shares, pos.yes_avg_cost
        else:
            held, avg = pos.no_shares, pos.no_avg_cost
        if held <= 0:
            return 0.0
        sell = min(shares, held)
        proceeds = sell * price
        pnl = (price - avg) * sell
        self.cash += proceeds
        if side == "YES":
            pos.yes_shares -= sell
            if pos.yes_shares <= 1e-12:
                pos.yes_shares = 0.0
                pos.yes_avg_cost = 0.0
        else:
            pos.no_shares -= sell
            if pos.no_shares <= 1e-12:
                pos.no_shares = 0.0
                pos.no_avg_cost = 0.0
        if pos.yes_shares <= 1e-12 and pos.no_shares <= 1e-12:
            del self.positions[market_id]
        else:
            self.positions[market_id] = pos
        return float(pnl)

    def settle(self, market_id: str, side: OutcomeSide, payout_per_share: float) -> tuple[float, float]:
        """Returns (cash_proceeds, settlement_pnl) for the given side."""
        pos = self.positions.get(market_id)
        if not pos:
            return 0.0, 0.0
        if side == "YES":
            shares, avg = pos.yes_shares, pos.yes_avg_cost
        else:
            shares, avg = pos.no_shares, pos.no_avg_cost
        if shares <= 0:
            return 0.0, 0.0
        proceeds = shares * payout_per_share
        pnl = proceeds - shares * avg
        self.cash += proceeds
        if side == "YES":
            pos.yes_shares = 0.0
            pos.yes_avg_cost = 0.0
        else:
            pos.no_shares = 0.0
            pos.no_avg_cost = 0.0
        if pos.yes_shares <= 1e-12 and pos.no_shares <= 1e-12:
            del self.positions[market_id]
        else:
            self.positions[market_id] = pos
        return float(proceeds), float(pnl)

    def has_any_shares(self, market_id: str) -> bool:
        pos = self.positions.get(market_id)
        if not pos:
            return False
        return pos.yes_shares > 0 or pos.no_shares > 0
