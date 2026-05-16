"""CLI: print latest (or specific) backtest report summary."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from btc5m_agents.backtest.engine import resolve_latest_run_dir
from btc5m_agents.config import get_settings

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    run_id: str = typer.Option("latest", "--run-id", help="'latest' symlink or a run folder name"),
    tail: int = typer.Option(8, "--tail", help="Rows of equity curve to show"),
) -> None:
    s = get_settings()
    base = s.resolved_reports_dir() / "backtests"
    if run_id == "latest":
        run_dir = resolve_latest_run_dir(s)
    else:
        run_dir = base / run_id
    if not run_dir.exists():
        console.print(f"[red]Run dir not found:[/red] {run_dir}")
        raise typer.Exit(code=1)

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        console.print(summary_path.read_text(encoding="utf-8"))

    eq_path = run_dir / "equity_curve.csv"
    if eq_path.exists():
        df = pd.read_csv(eq_path)
        console.print(f"\n[bold]Equity curve (last {tail})[/bold]")
        console.print(df.tail(tail).to_string(index=False))

    dec_path = run_dir / "decisions.jsonl"
    if dec_path.exists():
        console.print("\n[bold]Sample decisions (first 3 lines, truncated)[/bold]")
        with dec_path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                obj = json.loads(line)
                txt = json.dumps(obj, indent=2, default=str)
                console.print(txt[:2000] + ("..." if len(txt) > 2000 else ""))


if __name__ == "__main__":
    app()
