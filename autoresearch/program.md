# autoresearch

This is an experiment to have the agent do its own prompt research.

## Setup

To set up a new run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may27`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `main`.
3. **Read the in-scope files**:
   - `README.md` - project context.
   - `src/btc5m_agents/agents/schemas.py` - structured output contract.
   - `src/btc5m_agents/agents/graph.py` - where prompts are used.
   - `src/btc5m_agents/agents/prompts.py` - the only editable file.
   - `autoresearch/config.yaml` - fixed experiment config.
   - `autoresearch/baskets/manifest.json` - basket list used for eval.
4. **Verify baskets exist**: if `data/cache/baskets/*.parquet` are missing, run:
   - `python -m btc5m_agents.scripts.build_stratified_baskets --markets-per-basket 5`
5. **Initialize baseline**:
   - `python -m btc5m_agents.scripts.run_basket_eval > autoresearch/run.log 2>&1`
   - `python -m btc5m_agents.scripts.record_experiment --baseline -d "baseline prompts"`
6. **Confirm and go**: once baseline is logged, start experimentation.

## Experimentation

Each experiment runs a fixed basket harness. The harness evaluates all baskets in `autoresearch/baskets/manifest.json`.

**What you CAN do:**
- Modify only `src/btc5m_agents/agents/prompts.py`.
- Change prompt wording, examples, constraints, and calibration language.

**What you CANNOT do:**
- Modify `graph.py`, `schemas.py`, execution/sizing code, basket files, or manifest.
- Change evaluator settings (`decision_every_sec`, backend, model) mid-run.
- Use `--mock-llm` for official eval.
- Use `btc_5m_2s_60m.parquet` or smoke files for keep/discard decisions.

**Goal:** maximize mean basket score. Higher is better.

**Simplicity criterion:** all else equal, simpler prompts are better. Keep gains that are meaningful; prefer cleaner prompts over complex prompt bloat. If prompts get too long, context window limits will be exceeded.

**First run:** always baseline first, before any edits.

## Strategy-first policy

Do not run random prompt edits. Run data-driven strategy experiments.

For each experiment, use this flow:

1. Diagnose regime failures from prior results:
   - read latest `autoresearch/experiments.jsonl`
   - identify worst 1-2 baskets from `AUTORESEARCH_BASKET_SCORES`
   - open corresponding runs from `autoresearch/last_eval.json`
   - inspect:
     - `reports/backtests/<run_id>/summary.json`
     - `reports/backtests/<run_id>/trades.csv`
     - `reports/backtests/<run_id>/decisions.jsonl`
2. Form one explicit strategy hypothesis tied to observed behavior.
3. Implement that strategy in `prompts.py` with clear, testable rules.
4. Run full basket eval and compare to prior baseline/best.

Good strategy hypotheses include:
- tighten HOLD behavior in chop to reduce low-edge churn
- require stronger confidence/edge before directional buys in wide_spread
- de-risk faster near expiry in late_reversal
- rebalance buy vs sell behavior when exposure is elevated

Every experiment description must state:
- which basket(s) motivated the change
- what behavioral policy changed
- what outcome is expected (e.g. lower drawdown in chop, higher return in high_btc_move)

## Output format

`record_experiment` prints control lines:

```
AUTORESEARCH_STATUS=keep|discard|crash
AUTORESEARCH_MEAN_SCORE=...
AUTORESEARCH_BEST_MEAN_SCORE=...
AUTORESEARCH_BASKET_SCORES=name:score,...
AUTORESEARCH_REASON=...
```

## Logging results

Every experiment appends one JSON row to `autoresearch/experiments.jsonl` with:
- status (`keep` / `discard` / `crash`)
- mean score
- per-basket scores
- per-basket run IDs
- description of the hypothesis
- prompt hash

`prompts_best.py` and `best_mean_score.txt` are updated automatically on keep.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/may27`).

LOOP FOREVER:

1. Check git state and recent experiment history.
2. Diagnose weakest baskets and extract a strategy hypothesis.
3. Apply one focused strategy hypothesis to `src/btc5m_agents/agents/prompts.py`.
4. Commit the prompt change.
5. Run eval:
   - `python -m btc5m_agents.scripts.run_basket_eval > autoresearch/run.log 2>&1`
6. Score/log:
   - `python -m btc5m_agents.scripts.record_experiment -d "<short description>"`
7. If status is `keep`, continue from this commit.
8. If status is `discard` or `crash`, reset and restore:
   - `git reset --hard HEAD~1`
   - `cp autoresearch/prompts_best.py src/btc5m_agents/agents/prompts.py`

Scoring formula (per basket):

```
total_return_pct - 2 * max(0, max_drawdown_pct - 40)
```

Experiment score is the mean across all baskets in manifest.

## Crash handling

If eval crashes:
1. Inspect `autoresearch/run.log` (e.g. `tail -n 80 autoresearch/run.log`).
2. Fix obvious issues and retry once if appropriate.
3. If the idea is bad, log crash/discard and move on.

## JSON schema contract (strict)

Prompt outputs must satisfy:
- PriceView: `direction`, `confidence`, `signals`
- PolyViewLLM: `fair_prob_up`, `signals` (no `edge`)
- RiskView: `action`, `max_size`, `confidence` (`HOLD` requires `max_size: 0`)

Invalid keys can trigger parser fallback behavior and corrupt experiment quality.

## NEVER STOP

Once loop begins, do not pause to ask for permission to continue. Continue autonomously until user interrupts.
