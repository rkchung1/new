"""Replay engine: timestamp-ordered snapshots, LangGraph decisions, broker, settlement."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd
from btc5m_agents.agents.schemas import Decision
from btc5m_agents.backtest.broker import maybe_execute
from btc5m_agents.backtest.metrics import Summary, build_summary, summary_to_dict
from btc5m_agents.backtest.portfolio import Portfolio
from btc5m_agents.config import Settings, get_settings
from btc5m_agents.data.btc_5m_replay import load_btc_5m_replay
from btc5m_agents.features import FeatureEngine, MarketStateCache
from btc5m_agents.types import MarketPosition, PortfolioSnapshot, Snapshot, Trade


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _prompts_sha256(settings: Settings) -> Optional[str]:
    p = settings.project_root / "src" / "btc5m_agents" / "agents" / "prompts.py"
    if not p.exists():
        return None
    return _sha256_file(p)


def _update_latest_symlink(reports_backtests: Path, run_id: str) -> None:
    reports_backtests.mkdir(parents=True, exist_ok=True)
    link = reports_backtests / "latest"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(run_id, target_is_directory=True)
    except OSError:
        (reports_backtests / "latest_run_id.txt").write_text(run_id, encoding="utf-8")


def _resolve_run_dir(reports_backtests: Path, run_id: str) -> Path:
    if run_id == "latest":
        link = reports_backtests / "latest"
        if link.is_symlink() or link.is_dir():
            return link.resolve()
        txt = reports_backtests / "latest_run_id.txt"
        if txt.exists():
            rid = txt.read_text(encoding="utf-8").strip()
            return reports_backtests / rid
    return reports_backtests / run_id


def _optional_float(row: pd.Series, key: str) -> Optional[float]:
    if key not in row.index:
        return None
    v = row[key]
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return float(v)


def _optional_int(row: pd.Series, key: str) -> Optional[int]:
    if key not in row.index:
        return None
    v = row[key]
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return int(v)


def row_to_snapshot(row: pd.Series) -> Snapshot:
    """Agent-visible snapshot (no settlement / resolution peek)."""
    q = row.get("question")
    if q is None or (isinstance(q, float) and pd.isna(q)):
        q_out = None
    else:
        q_out = str(q)
    return Snapshot(
        ts=int(row["ts"]),
        market_id=str(row["market_id"]),
        yes_token_id=str(row["yes_token_id"]),
        event_slug=str(row["event_slug"]),
        yes_price=float(row["yes_price"]) if pd.notna(row["yes_price"]) else float("nan"),
        no_price=float(row["no_price"]) if "no_price" in row.index and pd.notna(row["no_price"]) else float("nan"),
        mins_to_expiry=float(row["mins_to_expiry"]),
        btc_price=float(row["btc_price"]) if pd.notna(row["btc_price"]) else float("nan"),
        btc_ret_5m=float(row["btc_ret_5m"]) if pd.notna(row["btc_ret_5m"]) else float("nan"),
        btc_vol_30m=float(row["btc_vol_30m"]) if pd.notna(row["btc_vol_30m"]) else float("nan"),
        question=q_out,
        elapsed_sec=_optional_int(row, "elapsed_sec"),
        secs_to_expiry=_optional_float(row, "secs_to_expiry"),
        bid_yes=_optional_float(row, "bid_yes"),
        ask_yes=_optional_float(row, "ask_yes"),
        bid_no=_optional_float(row, "bid_no"),
        ask_no=_optional_float(row, "ask_no"),
        btc_strike=_optional_float(row, "btc_strike"),
        btc_gap=_optional_float(row, "btc_gap"),
    )


def portfolio_to_snapshot(
    port: Portfolio,
    yes_marks: dict[str, float],
    no_marks: dict[str, float],
) -> PortfolioSnapshot:
    pos = {
        k: MarketPosition(
            market_id=v.market_id,
            yes_shares=v.yes_shares,
            yes_avg_cost=v.yes_avg_cost,
            no_shares=v.no_shares,
            no_avg_cost=v.no_avg_cost,
        )
        for k, v in port.positions.items()
    }
    return PortfolioSnapshot(
        cash=port.cash,
        positions=pos,
        exposure_usd=port.exposure_usd(yes_marks, no_marks),
    )


class BacktestEngine:
    def __init__(
        self,
        replay_path: Optional[Path],
        graph: Any,
        *,
        settings: Optional[Settings] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        decision_every_sec: Optional[int] = None,
        llm_backend: Optional[str] = None,
        llm_model: Optional[str] = None,
        vllm_base_url: Optional[str] = None,
    ) -> None:
        self._s = settings or get_settings()
        self._replay_path = replay_path or self._s.btc_5m_replay_path
        self._decision_every_sec = (
            decision_every_sec if decision_every_sec is not None else self._s.replay_decision_every_sec
        )
        self._llm_backend = llm_backend
        self._llm_model = llm_model
        self._vllm_base_url = vllm_base_url
        self._graph = graph
        df = load_btc_5m_replay(
            self._replay_path,
            settings=self._s,
            start_date=start_date,
            end_date=end_date,
            decision_every_sec=self._decision_every_sec,
        )
        self._df = df.sort_values(["ts", "market_id"]).reset_index(drop=True)
        self.portfolio = Portfolio(cash=self._s.initial_cash)
        self.last_yes_marks: dict[str, float] = {}
        self.last_no_marks: dict[str, float] = {}
        self.settled_markets: set[str] = set()
        self.trades: list[Trade] = []
        self.equity_rows: list[dict[str, Any]] = []
        self.decision_logs: list[dict[str, Any]] = []
        self.realized_pnl = 0.0
        self.settlement_pnl = 0.0
        self._market_win: dict[str, float] = {}  # cumulative pnl per market from sells+settle
        self._had_position_at_settle: set[str] = set()
        self._state_cache = MarketStateCache(history_sec=self._s.market_history_sec)
        self._feature_engine = FeatureEngine()

    def _maybe_settle(self, row: pd.Series) -> None:
        ts = int(row["ts"])
        mid = str(row["market_id"])
        end_ts = int(row["market_end_ts"])
        if ts < end_ts:
            return
        if mid in self.settled_markets:
            return
        sy = row.get("settlement_yes")
        if sy is None or (isinstance(sy, float) and pd.isna(sy)):
            if self.portfolio.has_any_shares(mid):
                return
            self.settled_markets.add(mid)
            self._state_cache.clear_market(mid)
            return
        payout_yes = float(sy)
        payout_no = 1.0 - payout_yes
        had_yes = self.portfolio.position_shares(mid, "YES") > 0
        had_no = self.portfolio.position_shares(mid, "NO") > 0
        if had_yes or had_no:
            self._had_position_at_settle.add(mid)
        if had_yes:
            proceeds, pnl = self.portfolio.settle(mid, "YES", payout_yes)
            self.settlement_pnl += pnl
            self._market_win[mid] = self._market_win.get(mid, 0.0) + pnl
            self.trades.append(
                Trade(
                    ts=ts,
                    market_id=mid,
                    action="SETTLE_YES",
                    shares=float("nan"),
                    price=float(payout_yes),
                    cost=float(-proceeds),
                    cash_after=float(self.portfolio.cash),
                )
            )
        if had_no:
            proceeds, pnl = self.portfolio.settle(mid, "NO", payout_no)
            self.settlement_pnl += pnl
            self._market_win[mid] = self._market_win.get(mid, 0.0) + pnl
            self.trades.append(
                Trade(
                    ts=ts,
                    market_id=mid,
                    action="SETTLE_NO",
                    shares=float("nan"),
                    price=float(payout_no),
                    cost=float(-proceeds),
                    cash_after=float(self.portfolio.cash),
                )
            )
        self.settled_markets.add(mid)
        self._state_cache.clear_market(mid)

    def run(self) -> Tuple[Path, Summary]:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        reports = self._s.resolved_reports_dir() / "backtests" / run_id
        reports.mkdir(parents=True, exist_ok=True)

        peak_equity = self._s.initial_cash

        for _, row in self._df.iterrows():
            mid = str(row["market_id"])
            ts = int(row["ts"])
            end_ts = int(row["market_end_ts"])
            yp = float(row["yes_price"]) if pd.notna(row["yes_price"]) else float("nan")
            np_ = float(row["no_price"]) if "no_price" in row.index and pd.notna(row["no_price"]) else float("nan")
            bid_yes = _optional_float(row, "bid_yes")
            ask_yes = _optional_float(row, "ask_yes")
            bid_no = _optional_float(row, "bid_no")
            ask_no = _optional_float(row, "ask_no")
            if yp == yp:
                self.last_yes_marks[mid] = yp
            if np_ == np_:
                self.last_no_marks[mid] = np_

            yes_marks = {k: v for k, v in self.last_yes_marks.items() if v == v}
            no_marks = {k: v for k, v in self.last_no_marks.items() if v == v}
            trade: Optional[Trade] = None

            snap = row_to_snapshot(row)
            self._state_cache.update(snap)
            dynamics_f, prediction_f = self._feature_engine.compute(self._state_cache, snap)
            port_snap = portfolio_to_snapshot(self.portfolio, yes_marks, no_marks)

            if ts >= end_ts:
                self._maybe_settle(row)
                decision = Decision(
                    action="HOLD",
                    size_usd=0.0,
                    predicted_prob_up=float(snap.yes_price) if snap.yes_price == snap.yes_price else 0.5,
                    confidence=0.0,
                    rationale="expiry_snapshot_no_trading",
                )
                out = {
                    "price_view": None,
                    "poly_view": None,
                    "risk_view": None,
                    "decision": decision.model_dump(),
                }
            else:
                state: dict[str, Any] = {
                    "market_id": mid,
                    "ts": ts,
                    "dynamics_features": dynamics_f.model_dump(),
                    "prediction_market_features": prediction_f.model_dump(),
                    "portfolio": port_snap.model_dump(mode="json"),
                }
                out = self._graph.invoke(state)
                decision = Decision.model_validate(out["decision"])

                cur_mark_exp = self.portfolio.market_exposure_usd(mid, yes_marks, no_marks)
                tot_exp = self.portfolio.exposure_usd(yes_marks, no_marks)

                trade = maybe_execute(
                    decision,
                    ts=ts,
                    market_id=mid,
                    yes_price=yp,
                    no_price=np_,
                    bid_yes=bid_yes,
                    ask_yes=ask_yes,
                    bid_no=bid_no,
                    ask_no=ask_no,
                    cash=self.portfolio.cash,
                    yes_shares=self.portfolio.position_shares(mid, "YES"),
                    no_shares=self.portfolio.position_shares(mid, "NO"),
                    current_mark_exposure_usd=cur_mark_exp,
                    total_exposure_usd=tot_exp,
                    settings=self._s,
                )
                if trade and trade.action == "BUY_YES":
                    self.portfolio.apply_buy(mid, "YES", trade.shares, trade.price)
                    self.trades.append(trade)
                elif trade and trade.action == "SELL_YES":
                    pnl = self.portfolio.apply_sell(mid, "YES", trade.shares, trade.price)
                    self.realized_pnl += pnl
                    self._market_win[mid] = self._market_win.get(mid, 0.0) + pnl
                    self.trades.append(trade)
                elif trade and trade.action == "BUY_NO":
                    self.portfolio.apply_buy(mid, "NO", trade.shares, trade.price)
                    self.trades.append(trade)
                elif trade and trade.action == "SELL_NO":
                    pnl = self.portfolio.apply_sell(mid, "NO", trade.shares, trade.price)
                    self.realized_pnl += pnl
                    self._market_win[mid] = self._market_win.get(mid, 0.0) + pnl
                    self.trades.append(trade)

            mv = self.portfolio.market_value(yes_marks, no_marks)
            eq = self.portfolio.cash + mv
            peak_equity = max(peak_equity, eq)
            dd = (peak_equity - eq) / peak_equity * 100 if peak_equity else 0.0

            self.equity_rows.append(
                {
                    "ts": ts,
                    "cash": self.portfolio.cash,
                    "positions_value": mv,
                    "total_equity": eq,
                    "drawdown_pct": float(dd),
                }
            )

            log = {
                "ts": ts,
                "market_id": mid,
                "market_end_ts": end_ts,
                "dynamics_features": dynamics_f.model_dump(),
                "prediction_market_features": prediction_f.model_dump(),
                "portfolio_before": port_snap.model_dump(mode="json"),
                "price_view": out.get("price_view"),
                "poly_view": out.get("poly_view"),
                "risk_view": out.get("risk_view"),
                "decision": decision.model_dump(),
                "confidence": decision.confidence,
                "predicted_prob_up": decision.predicted_prob_up,
                "executed_trade": trade.model_dump() if trade else None,
            }
            sy = row.get("settlement_yes")
            if sy is not None and not (isinstance(sy, float) and pd.isna(sy)):
                log["settlement_y_for_metrics"] = float(sy)
            else:
                log["settlement_y_for_metrics"] = None
            self.decision_logs.append(log)

        equity_df = pd.DataFrame(self.equity_rows)
        trades_df = pd.DataFrame([t.model_dump() for t in self.trades])

        # Win rate: markets we traded (buy or sell) with positive combined pnl
        traded_markets = set()
        for t in self.trades:
            if t.action in ("BUY_YES", "SELL_YES", "BUY_NO", "SELL_NO"):
                traded_markets.add(t.market_id)
        wins = sum(1 for m in traded_markets if self._market_win.get(m, 0.0) > 0)
        win_rate = wins / len(traded_markets) if traded_markets else 0.0

        decisions_df = pd.DataFrame(self.decision_logs)

        summary = build_summary(
            initial_cash=self._s.initial_cash,
            equity_curve=equity_df,
            trades=trades_df,
            decisions=decisions_df,
            realized_pnl=self.realized_pnl,
            settlement_pnl=self.settlement_pnl,
            num_markets_traded=len(traded_markets),
            win_rate=float(win_rate),
        )

        # Persist
        trades_df.to_csv(reports / "trades.csv", index=False)
        equity_df.to_csv(reports / "equity_curve.csv", index=False)
        with (reports / "decisions.jsonl").open("w", encoding="utf-8") as f:
            for line in self.decision_logs:
                f.write(json.dumps(line, default=str) + "\n")
        (reports / "summary.json").write_text(
            json.dumps(summary_to_dict(summary), indent=2),
            encoding="utf-8",
        )

        src_path = self._replay_path
        if not src_path.is_absolute():
            src_path = self._s.project_root / src_path
        cfg_snap = {
            "replay_path": str(src_path),
            "replay_sha256": _sha256_file(src_path) if src_path.exists() else None,
            "replay_rows": len(self._df),
            "decision_every_sec": self._decision_every_sec,
            "initial_cash": self._s.initial_cash,
            "max_position_per_market_usd": self._s.max_position_per_market_usd,
            "max_total_exposure_usd": self._s.max_total_exposure_usd,
            "slippage": self._s.slippage,
            "price_floor": self._s.price_floor,
            "price_ceiling": self._s.price_ceiling,
            "llm_backend": self._llm_backend,
            "llm_model": self._llm_model,
            "vllm_base_url": self._vllm_base_url,
            "llm_timeout_sec": self._s.llm_timeout_sec,
            "llm_max_tokens": self._s.llm_max_tokens,
            "market_history_sec": self._s.market_history_sec,
            "prompts_sha256": _prompts_sha256(self._s),
        }
        (reports / "config_snapshot.json").write_text(json.dumps(cfg_snap, indent=2), encoding="utf-8")

        _update_latest_symlink(self._s.resolved_reports_dir() / "backtests", run_id)
        return reports, summary


def resolve_latest_run_dir(settings: Optional[Settings] = None) -> Path:
    s = settings or get_settings()
    return _resolve_run_dir(s.resolved_reports_dir() / "backtests", "latest")
