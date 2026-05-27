# BTC 5m Polymarket multi-agent backtester

Reproducible **offline backtesting** for Polymarket BTC Up or Down 5-minute markets using a pre-built **`btc_5m_2s.parquet`** replay file (~2 second ticks, strike-relative BTC, bid/ask). A small **LangGraph** workflow runs four typed agents with a simulated broker.

**There is no live trading and no authenticated Polymarket access.**

## What the system does

1. **Load** `data/cache/btc_5m_2s.parquet` (2s Polymarket + BTC strike-gap snapshots).
2. **Normalize** into an engine replay table: filter `elapsed >= 101` (data starts 100s into each 5m window), optional date filter, subsample agent steps (default every 10s), append synthetic expiry rows at `start_time + 300`.
3. **Backtest** by walking rows in global `(ts, market_id)` order, calling LangGraph each step, executing at **bid/ask** with slippage, and **settling** YES at expiry using `winner` (engine-only, not shown to agents).
4. **Report** trades, decisions, equity curve, and summary metrics under `reports/backtests/{run_id}/`.

### Dormant scripts (kept, not used by backtest)

- `discover_markets` — Gamma API market discovery
- `build_dataset` — 1-minute API replay builder (`replay_*.parquet`)

These may be revived later; the active path uses only `btc_5m_2s.parquet`.

## How the backtester works

- Rows are sorted by `(ts, market_id)`; `ts` comes from `timestamp_log` (= `start_time + elapsed`).
- **Trading** only when `ts < market_end_ts` and `elapsed >= 101`.
- **Settlement** on rows with `ts >= market_end_ts` (synthetic expiry row per market).
- **Execution:** BUY at `ask_YES + slippage`, SELL at `bid_YES - slippage`, clipped to `[0.01, 0.99]`.

### Defaults

| Setting | Default |
|--------|---------|
| Initial cash | $1,000 |
| Max position / market | $100 |
| Max total exposure | $300 |
| Slippage | $0.01 |
| Decision interval | 10 seconds |
| Data start (elapsed) | 101 seconds |

Override via `.env` / `Settings` in `config.py`.

## Avoiding lookahead bias

- Each tick appends to a **causal 15-minute cache** (global BTC tape + per-market book history), then a **FeatureEngine** emits compact `BtcFeatures` / `PolyFeatures` (no future rows).
- Agents receive **role-specific feature JSON** only—not full snapshots. **`winner` / `resolved` / `settlement_yes` are never in the LLM payload.**
- `settlement_yes` is used only by the engine at/after expiry for cash settlement and Brier metrics.
- Brier score uses pre-expiry rows only (`ts < market_end_ts`).

## LangGraph agents

| Agent | Role |
|-------|------|
| **Price Analyst** | `btc_features`: momentum, vol, trend, strike gap, timing |
| **Polymarket Analyst** | `poly_features`: mids, spread, imbalance, prob divergence |
| **Risk Manager** | Execution policy from enriched state (`pos`, `yes_sh`/`no_sh`, `mkt_exp`, `eq`, `buy_cap`, `sell_yes_cap`/`sell_no_cap`, `exp_pct`, `cash`, analyst views) → `action`, `max_size`, `confidence` |
| **Portfolio Manager** | Deterministic map from risk policy → `Decision` (no LLM) |

Use **`--mock-llm`** for deterministic runs without any LLM.

### LLM backends

| Backend | Config | Notes |
|---------|--------|--------|
| **openai** | `LLM_BACKEND=openai`, `OPENAI_API_KEY` | Default cloud API |
| **vllm** | `LLM_BACKEND=vllm`, `VLLM_BASE_URL=http://localhost:8000/v1`, `LLM_MAX_TOKENS=1024` | Local vLLM; raise `LLM_MAX_TOKENS` if structured JSON truncates |
| **mock** | `--mock-llm` | Rule-based agents, no HTTP |

## CLI

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Run backtest (mock agents):

```bash
python -m btc5m_agents.scripts.run_backtest --mock-llm --decision-every-sec 10
```

Filter by market `start_time` (UTC calendar days covered by the parquet):

```bash
python -m btc5m_agents.scripts.run_backtest --mock-llm \
  --start-date 2026-03-01 --end-date 2026-03-04
```

With OpenAI:

```bash
python -m btc5m_agents.scripts.run_backtest --llm-backend openai --model gpt-4o-mini
```

With local vLLM (OpenAI-compatible API on port 8000):

```bash
# Terminal 1: start vLLM
vllm serve <model> --port 8000

# Terminal 2: backtest (set VLLM_MODEL to the served model id)
python -m btc5m_agents.scripts.run_backtest \
  --llm-backend vllm \
  --model <model> \
  --decision-every-sec 60
```

Or via `.env`:

```env
LLM_BACKEND=vllm
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=<model>
VLLM_API_KEY=EMPTY
LLM_MAX_TOKENS=1024
MARKET_HISTORY_SEC=900
```

`decisions.jsonl` logs `btc_features` and `poly_features` per step (not full snapshots).

Inspect the latest run:

```bash
python -m btc5m_agents.scripts.inspect_results --run-id latest
```

## Replay subsets (fast testing)

Checked-in slices under `data/cache/` (regenerate from the full `btc_5m_2s.parquet`):

| File | Markets | Raw rows | Normalized (`--decision-every-sec 60`) |
|------|---------|----------|----------------------------------------|
| `btc_5m_2s_smoke.parquet` | 3 | 288 | ~15 steps (~36 LLM calls) |
| `btc_5m_2s_60m.parquet` | 60 consecutive (from file start) | 5,755 | ~300 steps (~900 LLM calls) |
| `btc_5m_2s_24m.parquet` | 24 consecutive (after 60m block, no overlap) | 2,135 | ~115 steps (~273 LLM calls) |

Build or customize:

```bash
python -m btc5m_agents.scripts.build_replay_subset --max-markets 3
python -m btc5m_agents.scripts.build_replay_subset --max-markets 60
python -m btc5m_agents.scripts.build_replay_subset --max-markets 24 --skip-markets 60
# explicit slugs:
python -m btc5m_agents.scripts.build_replay_subset --slug btc-updown-5m-1771847400 --slug btc-updown-5m-1771847700
```

Example backtest on the 60-market slice (~5 hours of consecutive 5m windows):

```bash
python -m btc5m_agents.scripts.run_backtest \
  --replay data/cache/btc_5m_2s_60m.parquet \
  --decision-every-sec 60 \
  --llm-backend vllm
```

## Data file

Place your replay at:

`data/cache/btc_5m_2s.parquet`

Expected columns: `slug`, `start_time`, `elapsed`, `ask_YES`, `bid_YES`, `timestamp_log`, `btc_strike`, `btc_current`, `btc_gap`, `winner` (plus optional `ask_NO`, `bid_NO`, `resolved`).

Normalized cache (optional, auto-written): `data/cache/btc_5m_2s_normalized_{hash}.parquet`

## Reports

Each run writes `trades.csv`, `decisions.jsonl`, `equity_curve.csv`, `summary.json`, and `config_snapshot.json` under `reports/backtests/{run_id}/`.

## Prompt autoresearch

Optimize agent prompts with a Cursor agent following [`autoresearch/program.md`](autoresearch/program.md). Each experiment runs **all** stratified baskets under `data/cache/baskets/` (see [`autoresearch/baskets/manifest.json`](autoresearch/baskets/manifest.json)); **keep/discard** uses the **mean** risk-adjusted score across baskets (not `btc_5m_2s_60m`).

```bash
# Build baskets once (from full replay)
python -m btc5m_agents.scripts.build_stratified_baskets --markets-per-basket 5

# Baseline
python -m btc5m_agents.scripts.run_basket_eval > autoresearch/run.log 2>&1
python -m btc5m_agents.scripts.record_experiment --baseline -d "baseline"

# After editing prompts.py — one experiment
python -m btc5m_agents.scripts.run_basket_eval > autoresearch/run.log 2>&1
python -m btc5m_agents.scripts.record_experiment -d "your hypothesis"
```

Start Cursor with: *Read autoresearch/program.md and set up a new prompt autoresearch run. Then start the experiment loop.*

## Disclaimer

This repository is for **research and education**. Prediction markets involve financial risk; this code does not place orders or move funds.
