"""Score a basket eval (mean across baskets) and keep or discard prompt changes."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

from btc5m_agents.autoresearch.config import load_autoresearch_config, resolve_path
from btc5m_agents.autoresearch.paths import (
    best_mean_score_path,
    experiments_path,
    last_eval_path,
    prompts_best_path,
    prompts_path,
    prompts_sha256,
)
from btc5m_agents.autoresearch.scorer import mean_score, score_summary
from btc5m_agents.config import get_settings

app = typer.Typer(add_completion=False)
console = Console()


def _load_summary(reports_dir: Path) -> dict[str, Any]:
    path = reports_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary.json in {reports_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_best_score(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return float(text.splitlines()[0])


def _write_best_score(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{value}\n", encoding="utf-8")


def _append_experiment(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


@app.command()
def main(
    description: str = typer.Option("", "--description", "-d", help="What this experiment tried"),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Autoresearch config YAML",
    ),
    eval_file: Optional[Path] = typer.Option(
        None,
        "--eval-file",
        help="Path to last_eval.json (default: autoresearch/last_eval.json)",
    ),
    baseline: bool = typer.Option(
        False,
        "--baseline",
        help="Set current mean score as best (first run)",
    ),
) -> None:
    load_dotenv()
    s = get_settings()
    cfg_path = resolve_path(config or Path("autoresearch/config.yaml"), s)
    cfg = load_autoresearch_config(cfg_path, s)
    dd_limit = float(cfg.get("max_drawdown_pct_limit", 40.0))
    dd_penalty = float(cfg.get("dd_penalty", 2.0))

    eval_path = eval_file or last_eval_path(s)
    if not eval_path.is_absolute():
        eval_path = s.project_root / eval_path
    if not eval_path.exists():
        console.print(f"[red]Eval file not found:[/red] {eval_path}")
        console.print("Run: python -m btc5m_agents.scripts.run_basket_eval")
        raise typer.Exit(code=1)

    eval_doc = json.loads(eval_path.read_text(encoding="utf-8"))
    basket_entries = eval_doc.get("baskets") or {}
    if not basket_entries:
        console.print("[red]No baskets in eval file.[/red]")
        raise typer.Exit(code=1)

    pp = prompts_path(s)
    p_sha = prompts_sha256(pp) if pp.exists() else None

    basket_scores: dict[str, float] = {}
    basket_summaries: dict[str, dict[str, Any]] = {}
    basket_run_ids: dict[str, str] = {}
    crashed: list[str] = []

    for name, info in basket_entries.items():
        if info.get("status") != "ok" or not info.get("run_id"):
            crashed.append(name)
            continue
        run_id = str(info["run_id"])
        reports = s.resolved_reports_dir() / "backtests" / run_id
        summary = _load_summary(reports)
        sc = score_summary(summary, max_drawdown_pct_limit=dd_limit, dd_penalty=dd_penalty)
        basket_scores[name] = sc
        basket_run_ids[name] = run_id
        basket_summaries[name] = {
            "total_return_pct": summary.get("total_return_pct"),
            "max_drawdown_pct": summary.get("max_drawdown_pct"),
            "num_trades": summary.get("num_trades"),
            "brier_score": summary.get("brier_score"),
        }

    if crashed:
        console.print(f"[red]Crashed/missing baskets:[/red] {', '.join(crashed)}")
        _append_experiment(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "description": description,
                "status": "crash",
                "prompts_sha256": p_sha,
                "crashed_baskets": crashed,
                "basket_scores": basket_scores,
            },
            experiments_path(s),
        )
        print("AUTORESEARCH_STATUS=crash")
        raise typer.Exit(code=1)

    if len(basket_scores) != len(basket_entries):
        console.print("[red]Incomplete basket scores.[/red]")
        raise typer.Exit(code=1)

    experiment_score = mean_score(basket_scores)
    best_path = best_mean_score_path(s)
    prev_best = _read_best_score(best_path)

    if baseline or prev_best is None:
        status = "keep"
        reason = "baseline"
    elif experiment_score > prev_best:
        status = "keep"
        reason = "improved"
    else:
        status = "discard"
        reason = "no_improvement"

    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "description": description,
        "status": status,
        "reason": reason,
        "mean_score": experiment_score,
        "best_mean_score_before": prev_best,
        "prompts_sha256": p_sha,
        "basket_scores": basket_scores,
        "basket_run_ids": basket_run_ids,
        "basket_summaries": basket_summaries,
        "eval_file": str(eval_path.name),
    }
    _append_experiment(row, experiments_path(s))

    if status == "keep":
        _write_best_score(best_path, experiment_score)
        best_dest = prompts_best_path(s)
        shutil.copy2(pp, best_dest)

    scores_csv = ",".join(f"{k}:{v:.4f}" for k, v in sorted(basket_scores.items()))
    print(f"AUTORESEARCH_STATUS={status}")
    print(f"AUTORESEARCH_MEAN_SCORE={experiment_score:.6f}")
    best_display = experiment_score if status == "keep" else prev_best
    print(
        f"AUTORESEARCH_BEST_MEAN_SCORE={best_display:.6f}"
        if best_display is not None
        else "AUTORESEARCH_BEST_MEAN_SCORE=",
    )
    print(f"AUTORESEARCH_BASKET_SCORES={scores_csv}")
    print(f"AUTORESEARCH_REASON={reason}")

    console.print(f"[bold]{status.upper()}[/bold] mean={experiment_score:.4f} (best was {prev_best})")


if __name__ == "__main__":
    app()
