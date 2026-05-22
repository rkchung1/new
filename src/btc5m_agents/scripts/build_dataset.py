"""CLI: build replay Parquet from cached markets + CLOB + BTC.

Dormant: not used by run_backtest. Retained for possible future API-based replay.
"""

from __future__ import annotations

import typer
from rich.console import Console

from btc5m_agents.data.dataset_builder import build_replay_dataset

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
) -> None:
    path = build_replay_dataset(start_date, end_date)
    console.print(f"[green]Wrote[/green] {path}")


if __name__ == "__main__":
    app()
