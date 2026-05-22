"""Rolling market state caches (causal, up to 15 minutes)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from btc5m_agents.types import Snapshot


@dataclass(frozen=True)
class BtcPoint:
    ts: int
    price: float


@dataclass(frozen=True)
class BookTick:
    ts: int
    market_id: str
    yes_mid: float
    no_mid: float
    bid_yes: Optional[float]
    ask_yes: Optional[float]
    bid_no: Optional[float]
    ask_no: Optional[float]
    elapsed_sec: Optional[int]
    secs_to_expiry: Optional[float]
    btc_strike: Optional[float]
    btc_gap: Optional[float]


class BtcTapeCache:
    """Global BTC spot tape across all markets (deduped by ts)."""

    def __init__(self, history_sec: int = 900) -> None:
        self.history_sec = history_sec
        self._points: Deque[BtcPoint] = deque()
        self._by_ts: dict[int, float] = {}

    def update(self, snap: Snapshot) -> None:
        price = snap.btc_price
        if price != price:
            return
        if snap.ts not in self._by_ts:
            self._points.append(BtcPoint(ts=snap.ts, price=float(price)))
            self._by_ts[snap.ts] = float(price)
        self._prune(snap.ts)

    def _prune(self, current_ts: int) -> None:
        cutoff = current_ts - self.history_sec
        while self._points and self._points[0].ts < cutoff:
            old = self._points.popleft()
            self._by_ts.pop(old.ts, None)

    def series(self, current_ts: int) -> list[BtcPoint]:
        """Points with ts <= current_ts within history window."""
        cutoff = current_ts - self.history_sec
        return [p for p in self._points if cutoff <= p.ts <= current_ts]

    def price_at_lag(self, current_ts: int, lag_sec: int) -> Optional[float]:
        target = current_ts - lag_sec
        best: Optional[float] = None
        best_ts = -1
        for p in self._points:
            if p.ts <= target and p.ts > best_ts:
                best_ts = p.ts
                best = p.price
        return best


class MarketBookCache:
    """Per-market order book / strike history."""

    def __init__(self, market_id: str, history_sec: int = 900) -> None:
        self.market_id = market_id
        self.history_sec = history_sec
        self._ticks: Deque[BookTick] = deque()

    def update(self, snap: Snapshot) -> None:
        tick = BookTick(
            ts=snap.ts,
            market_id=snap.market_id,
            yes_mid=float(snap.yes_price) if snap.yes_price == snap.yes_price else 0.5,
            no_mid=float(snap.no_price) if snap.no_price == snap.no_price else 0.5,
            bid_yes=snap.bid_yes,
            ask_yes=snap.ask_yes,
            bid_no=snap.bid_no,
            ask_no=snap.ask_no,
            elapsed_sec=snap.elapsed_sec,
            secs_to_expiry=snap.secs_to_expiry,
            btc_strike=snap.btc_strike,
            btc_gap=snap.btc_gap,
        )
        if self._ticks and self._ticks[-1].ts == tick.ts:
            self._ticks[-1] = tick
        else:
            self._ticks.append(tick)
        self._prune(snap.ts)

    def _prune(self, current_ts: int) -> None:
        cutoff = current_ts - self.history_sec
        while self._ticks and self._ticks[0].ts < cutoff:
            self._ticks.popleft()

    def ticks(self, current_ts: int) -> list[BookTick]:
        cutoff = current_ts - self.history_sec
        return [t for t in self._ticks if cutoff <= t.ts <= current_ts]

    def value_at_lag(
        self,
        current_ts: int,
        lag_sec: int,
        getter: str,
    ) -> Optional[float]:
        target = current_ts - lag_sec
        best: Optional[BookTick] = None
        for t in self._ticks:
            if t.ts <= target and (best is None or t.ts > best.ts):
                best = t
        if best is None:
            return None
        return getattr(best, getter, None)  # type: ignore[no-any-return]


class MarketStateCache:
    """Facade: global BTC tape + per-market book caches."""

    def __init__(self, history_sec: int = 900) -> None:
        self.history_sec = history_sec
        self.btc_tape = BtcTapeCache(history_sec=history_sec)
        self._books: dict[str, MarketBookCache] = {}

    def book(self, market_id: str) -> MarketBookCache:
        if market_id not in self._books:
            self._books[market_id] = MarketBookCache(market_id, self.history_sec)
        return self._books[market_id]

    def update(self, snap: Snapshot) -> None:
        self.btc_tape.update(snap)
        self.book(snap.market_id).update(snap)

    def clear_market(self, market_id: str) -> None:
        self._books.pop(market_id, None)
