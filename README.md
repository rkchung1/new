# BTC 5m Polymarket multi-agent backtester

Reproducible **offline backtesting** for Polymarket [BTC Up or Down 5m](https://polymarket.com/event/btc-updown-5m-1778120400)-style markets: discover markets with the **Gamma API**, pull historical **YES** prices from the public **CLOB `/prices-history`**, align **BTC-USD 5m** candles from **Coinbase Exchange**, cache everything as **Parquet**, then replay **1-minute snapshots** through a small **LangGraph** workflow with four typed agents and a simulated broker.

**There is no live trading and no authenticated Polymarket access.**

## What the system does

1. **Discover** candidate event slugs `btc-updown-5m-{unix_start}` in a date range and resolve them via `GET https://gamma-api.polymarket.com/events?slug=...`.
2. **Ingest** YES token price history from `https://clob.polymarket.com/prices-history` and BTC candles from `https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=300`.
3. **Cache** raw pulls under `data/cache/` as Parquet (idempotent; safe to re-run).
4. **Build** a single **replay dataset** (`replay_{start}_{end}.parquet`) where each row is one `(timestamp, market)` snapshot with joined BTC context.
5. **Backtest** by walking rows in global timestamp order, calling a LangGraph graph each step, applying **conservative execution** (±$0.01 slippage, prices clipped to `[0.01, 0.99]`), enforcing **per-market** and **total** exposure caps, and **settling** YES shares at market end using the **resolved Polymarket outcome** from Gamma (`settlement_yes`).
6. **Report** trades, decisions, equity curve, and summary metrics under `reports/backtests/{run_id}/`.

## How the backtester works

- The engine sorts the replay table by `(ts, market_id)` and processes row-by-row.
- **Settlement** runs on the terminal snapshot at `ts == market_end_ts` (appended during dataset build) so cash flows occur exactly at expiry without lookahead during the window.
- On expiry rows the workflow is **skipped** (forced `HOLD`): no trading after the official window end.
- **Execution** uses `buy_price = clip(yes + 0.01)` and `sell_price = clip(yes - 0.01)` within `[0.01, 0.99]`, then converts `size_usd` to shares.

### Defaults

| Setting | Default |
|--------|---------|
| Initial cash | $1,000 |
| Max position / market | $100 |
| Max total exposure | $300 |
| Slippage | $0.01 |

Override via environment variables in `.env` (see `config.py` / `Settings`).

## Avoiding lookahead bias

- Agents receive a **`Snapshot`** built only from columns knowable at that `ts` (YES price, time to expiry, BTC features). **Settlement / resolution is never included** in the LLM payload.
- CLOB history fetches are **truncated to the market window** (plus one minute of padding for the last tick), so future prices for that market are not in cache.
- The replay file may contain `settlement_yes` for offline metrics and settlement accounting; the engine **strips it from the agent view** and only uses it after expiry for cash settlement.
- **Brier score** and **average confidence** metrics only use pre-expiry rows (`ts < market_end_ts`) so terminal bookkeeping rows do not pollute scoring.

## LangGraph agents (TradingAgents-style mapping)

| Agent | Role | TradingAgents analogy |
|-------|------|------------------------|
| **Price Analyst** | Interprets BTC momentum/vol from the snapshot | Market / fundamentals analyst |
| **Polymarket Analyst** | Reads implied probability from YES price vs fundamentals | Sentiment / news analyst |
| **Risk Manager** | Enforces exposure and dollar caps | Risk debator / compliance |
| **Portfolio Manager** | Emits `BUY_YES`, `SELL_YES`, or `HOLD` with size | Trader / portfolio manager |

Each agent returns a **Pydantic** schema (`agents/schemas.py`). Use **`--mock-llm`** for deterministic, API-free runs; otherwise configure `OPENAI_API_KEY` and pass **`--model`**.

## CLI

Install (editable):

```bash
cd new_polyagents
python3 -m venv .venv && source .venv/bin/activate  # Python 3.9+
pip install -e .
```

Discover markets (writes `data/cache/markets/markets_{start}_{end}.parquet`):

```bash
python -m btc5m_agents.scripts.discover_markets --start-date 2026-05-01 --end-date 2026-05-02
```

Build replay dataset:

```bash
python -m btc5m_agents.scripts.build_dataset --start-date 2026-05-01 --end-date 2026-05-02
```

Run backtest (mock agents, no OpenAI key required):

```bash
python -m btc5m_agents.scripts.run_backtest --mock-llm --start-date 2026-05-01 --end-date 2026-05-02
```

Run with OpenAI (requires `.env`):

```bash
python -m btc5m_agents.scripts.run_backtest --model gpt-4o-mini --start-date 2026-05-01 --end-date 2026-05-02
```

If `--start-date` / `--end-date` are omitted, the runner picks the **newest** `replay_*.parquet` in `data/cache/`.

Inspect the latest run:

```bash
python -m btc5m_agents.scripts.inspect_results --run-id latest
```

## Reports

Each run writes:

- `trades.csv` — includes `SETTLE` rows for expiry cashflows
- `decisions.jsonl` — full structured agent outputs + `settlement_y_for_metrics` (never fed to agents)
- `equity_curve.csv`
- `summary.json` — `initial_cash`, `final_equity`, `total_return_pct`, `max_drawdown_pct`, `num_trades`, `num_markets_traded`, `win_rate`, `realized_pnl`, `settlement_pnl`, `avg_agent_confidence`, `brier_score` (when labels exist)
- `config_snapshot.json` — settings + SHA-256 of the replay file

`reports/backtests/latest` symlink (or `latest_run_id.txt` fallback) points at the most recent run.

## Project layout

```
src/btc5m_agents/
  config.py
  types.py
  data/
  agents/
  backtest/
  scripts/
```

## Disclaimer

This repository is for **research and education**. Prediction markets involve financial risk; this code does not place orders or move funds.
