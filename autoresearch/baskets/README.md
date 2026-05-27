# Stratified replay baskets

Built from `data/cache/btc_5m_2s.parquet` via:

```bash
python -m btc5m_agents.scripts.build_stratified_baskets --markets-per-basket 5
python -m btc5m_agents.scripts.build_stratified_baskets --validate-only
```

| Basket | Criteria (trading window, elapsed ≥ 101) |
|--------|------------------------------------------|
| `yes_settled` | `winner` = UP; top \|btc_gap\| among UP markets |
| `no_settled` | `winner` = DOWN; top \|btc_gap\| among DOWN markets |
| `high_btc_move` | max \|btc_gap\| ≥ universe p90 |
| `chop` | yes_mid_std ≤ p25 and max \|btc_gap\| ≤ p25 |
| `late_reversal` | \|btc_gap@late − btc_gap@early\| ≥ p90 |
| `wide_spread` | mean(ask_YES − bid_YES) ≥ p90 |

Parquets: `data/cache/baskets/basket_*.parquet`. Metadata: `manifest.json`.

Backtest one basket:

```bash
python -m btc5m_agents.scripts.run_backtest \
  --replay data/cache/baskets/basket_chop.parquet \
  --decision-every-sec 60
```

**Autoresearch** runs all baskets and scores the mean — see [`../program.md`](../program.md):

```bash
python -m btc5m_agents.scripts.run_basket_eval
python -m btc5m_agents.scripts.record_experiment -d "description"
```
