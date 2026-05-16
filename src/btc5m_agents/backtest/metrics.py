"""Backtest summary metrics (pure functions + Summary model)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class Summary(BaseModel):
    initial_cash: float
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    num_trades: int
    num_markets_traded: int
    win_rate: float
    realized_pnl: float
    settlement_pnl: float
    avg_agent_confidence: float
    brier_score: Optional[float] = None


def max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    if not dd.notna().any():
        return 0.0
    return max(0.0, float(-dd.min() * 100))


def compute_brier(decisions: pd.DataFrame) -> Optional[float]:
    if decisions.empty or "predicted_prob_up" not in decisions.columns:
        return None
    if "settlement_y_for_metrics" not in decisions.columns:
        return None
    sub = decisions.dropna(subset=["settlement_y_for_metrics", "predicted_prob_up"])
    if sub.empty:
        return None
    if "ts" in sub.columns and "market_end_ts" in sub.columns:
        sub = sub[sub["ts"] < sub["market_end_ts"]]
    if sub.empty:
        return None
    y = sub["settlement_y_for_metrics"].astype(float)
    p = sub["predicted_prob_up"].astype(float)
    return float(np.mean((p - y) ** 2))


def build_summary(
    *,
    initial_cash: float,
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    decisions: pd.DataFrame,
    realized_pnl: float,
    settlement_pnl: float,
    num_markets_traded: int,
    win_rate: float,
) -> Summary:
    final_eq = float(equity_curve["total_equity"].iloc[-1]) if not equity_curve.empty else initial_cash
    ret_pct = (final_eq / initial_cash - 1.0) * 100 if initial_cash else 0.0
    mdd = max_drawdown_pct(equity_curve["total_equity"]) if not equity_curve.empty else 0.0
    n_tr = int(len(trades[trades["action"].isin(["BUY_YES", "SELL_YES"])])) if not trades.empty else 0
    avg_conf = 0.0
    if not decisions.empty and "confidence" in decisions.columns:
        dc = decisions
        if "ts" in dc.columns and "market_end_ts" in dc.columns:
            dc = dc[dc["ts"] < dc["market_end_ts"]]
        if not dc.empty:
            avg_conf = float(dc["confidence"].mean())
    brier = compute_brier(decisions)
    return Summary(
        initial_cash=initial_cash,
        final_equity=final_eq,
        total_return_pct=ret_pct,
        max_drawdown_pct=mdd,
        num_trades=n_tr,
        num_markets_traded=num_markets_traded,
        win_rate=win_rate,
        realized_pnl=realized_pnl,
        settlement_pnl=settlement_pnl,
        avg_agent_confidence=avg_conf,
        brier_score=brier,
    )


def summary_to_dict(s: Summary) -> dict[str, Any]:
    d = s.model_dump()
    # JSON-friendly
    for k, v in list(d.items()):
        if v is not None and isinstance(v, (np.floating, float)):
            d[k] = float(v)
    return d
