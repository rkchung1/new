"""Unit checks for derived poly edge."""

from __future__ import annotations

from btc5m_agents.agents.pricing import build_poly_view, compute_edge, poly_view_from_llm
from btc5m_agents.agents.schemas import PolyViewLLM


def test_compute_edge_matches_fair_minus_mid() -> None:
    assert compute_edge(0.68, 0.365) == 0.315
    assert compute_edge(0.415, 0.415) == 0.0


def test_compute_edge_clips_extremes() -> None:
    assert compute_edge(1.5, -0.2) == round(0.99 - 0.01, 4)


def test_build_poly_view_derives_edge() -> None:
    pv = build_poly_view(0.71, ["yes_cheap"], 0.65)
    assert pv.fair_prob_up == 0.71
    assert pv.edge == 0.06
    assert pv.signals == ["yes_cheap"]


def test_poly_view_from_llm() -> None:
    raw = PolyViewLLM(fair_prob_up=0.62, signals=["late_repricing"])
    pv = poly_view_from_llm(raw, 0.385)
    assert pv.edge == compute_edge(0.62, 0.385)
    assert pv.fair_prob_up == 0.62
