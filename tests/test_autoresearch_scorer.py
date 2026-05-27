"""Tests for autoresearch scoring."""

from __future__ import annotations

import unittest

from btc5m_agents.autoresearch.scorer import mean_score, score_summary


class TestAutoresearchScorer(unittest.TestCase):
    def test_score_penalizes_drawdown(self) -> None:
        good = score_summary({"total_return_pct": 10.0, "max_drawdown_pct": 30.0})
        bad_dd = score_summary({"total_return_pct": 10.0, "max_drawdown_pct": 50.0})
        self.assertGreater(good, bad_dd)
        self.assertAlmostEqual(bad_dd, 10.0 - 2.0 * 10.0)

    def test_mean_score(self) -> None:
        m = mean_score({"a": 1.0, "b": 3.0})
        self.assertAlmostEqual(m, 2.0)


if __name__ == "__main__":
    unittest.main()
