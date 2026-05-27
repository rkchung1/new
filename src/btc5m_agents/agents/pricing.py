"""Deterministic pricing helpers — edge is always derived from fair vs market mid."""

from __future__ import annotations

from btc5m_agents.agents.schemas import PolyView, PolyViewLLM

_PROB_FLOOR = 0.01
_PROB_CEILING = 0.99


def clip_prob(p: float) -> float:
    """Clamp probability to (0, 1) for stable edge arithmetic."""
    return float(max(_PROB_FLOOR, min(_PROB_CEILING, p)))


def compute_edge(fair_prob_up: float, yes_mid: float) -> float:
    """YES edge vs mid: positive means YES is cheap relative to fair."""
    fair = clip_prob(fair_prob_up)
    mkt = clip_prob(yes_mid)
    return round(fair - mkt, 4)


def build_poly_view(
    fair_prob_up: float,
    signals: list[str],
    yes_mid: float,
) -> PolyView:
    """Assemble PolyView with edge derived from fair_prob_up and yes_mid."""
    fair = clip_prob(fair_prob_up)
    return PolyView(
        fair_prob_up=fair,
        edge=compute_edge(fair, yes_mid),
        signals=signals[:4],
    )


def poly_view_from_llm(raw: PolyViewLLM, yes_mid: float) -> PolyView:
    """Convert LLM poly output (no edge field) into a full PolyView."""
    return build_poly_view(raw.fair_prob_up, raw.signals, yes_mid)
