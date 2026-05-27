"""Risk-adjusted backtest scoring for prompt autoresearch."""

from __future__ import annotations

from typing import Mapping


def score_summary(
    summary: Mapping[str, float | int | None],
    *,
    max_drawdown_pct_limit: float = 40.0,
    dd_penalty: float = 2.0,
) -> float:
    """Higher is better: total_return_pct minus penalty for excess drawdown."""
    ret = float(summary.get("total_return_pct") or 0.0)
    dd = float(summary.get("max_drawdown_pct") or 0.0)
    excess = max(0.0, dd - max_drawdown_pct_limit)
    return ret - dd_penalty * excess


def mean_score(basket_scores: Mapping[str, float]) -> float:
    if not basket_scores:
        return 0.0
    vals = list(basket_scores.values())
    return sum(vals) / len(vals)
