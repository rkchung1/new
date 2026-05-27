"""Tests for stratified basket labeling and validation."""

from __future__ import annotations

import unittest

import pandas as pd

from btc5m_agents.data.stratified_baskets import (
    BasketSpec,
    assign_baskets,
    basket_eligible,
    compute_market_stats,
    compute_thresholds,
    default_basket_specs,
    validate_assignments,
)


def _synthetic_replay() -> pd.DataFrame:
    """Six slugs with exaggerated traits for bucket tests."""
    rows = []
    specs = [
        ("up-big", "up", 300.0, 0.05, 0.01, 200.0),
        ("down-big", "down", 280.0, 0.06, 0.01, 180.0),
        ("chop-1", "up", 30.0, 0.02, 0.005, 5.0),
        ("chop-2", "down", 25.0, 0.025, 0.006, 4.0),
        ("late-1", "up", 120.0, 0.10, 0.02, 250.0),
        ("wide-1", "down", 100.0, 0.12, 0.05, 20.0),
    ]
    for slug, winner, gap_scale, mid_std, spread, late_delta in specs:
        for elapsed in [101, 150, 200, 260]:
            gap = gap_scale if elapsed < 200 else gap_scale * 0.5
            if slug == "late-1" and elapsed >= 260:
                gap = gap_scale * 2.0
            yes_mid = 0.5 + (mid_std if elapsed % 2 == 0 else -mid_std)
            rows.append(
                {
                    "slug": slug,
                    "start_time": 1_700_000_000,
                    "elapsed": elapsed,
                    "ask_YES": yes_mid + spread / 2,
                    "bid_YES": yes_mid - spread / 2,
                    "ask_NO": 0.5,
                    "bid_NO": 0.49,
                    "btc_strike": 100.0,
                    "btc_current": 100.0 + gap,
                    "btc_gap": gap if slug != "late-1" else (gap if elapsed < 260 else gap + late_delta),
                    "timestamp_log": 1_700_000_000 + elapsed,
                    "resolved": True,
                    "winner": winner,
                },
            )
    return pd.DataFrame(rows)


class TestStratifiedBaskets(unittest.TestCase):
    def test_assign_and_validate_disjoint(self) -> None:
        df = _synthetic_replay()
        stats = compute_market_stats(df)
        thresholds = compute_thresholds(stats)
        specs = default_basket_specs()
        assignments = assign_baskets(stats, thresholds, specs, markets_per_basket=1)
        report = validate_assignments(stats, assignments, specs, thresholds)
        self.assertTrue(report["all_passed"])
        all_slugs = [s for sl in assignments.values() for s in sl]
        self.assertEqual(len(all_slugs), len(set(all_slugs)))

    def test_yes_no_mutually_exclusive(self) -> None:
        df = _synthetic_replay()
        stats = compute_market_stats(df)
        t = compute_thresholds(stats)
        yes = BasketSpec("yes", "yes_settled", "", "y.parquet")
        no = BasketSpec("no", "no_settled", "", "n.parquet")
        for m in stats:
            if m.winner == "up":
                self.assertTrue(basket_eligible(yes, m, t))
                self.assertFalse(basket_eligible(no, m, t))
            elif m.winner == "down":
                self.assertTrue(basket_eligible(no, m, t))
                self.assertFalse(basket_eligible(yes, m, t))


if __name__ == "__main__":
    unittest.main()
