"""CLI: discover BTC 5m markets via Gamma slug enumeration.

Dormant: not used by run_backtest. Retained for possible future API-based replay.
"""

from __future__ import annotations

import typer
from rich.console import Console

from btc5m_agents.data.market_discovery import discover

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    start_date: str = typer.Option(..., "--start-date", help="YYYY-MM-DD (UTC day)"),
    end_date: str = typer.Option(..., "--end-date", help="YYYY-MM-DD (UTC day)"),
) -> None:
    path = discover(start_date, end_date)
    console.print(f"[green]Wrote[/green] {path}")


if __name__ == "__main__":
    app()
