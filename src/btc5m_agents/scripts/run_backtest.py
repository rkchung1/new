"""CLI: run LangGraph backtest on a replay dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from btc5m_agents.agents.graph import build_graph
from btc5m_agents.backtest.engine import BacktestEngine
from btc5m_agents.config import get_settings

app = typer.Typer(add_completion=False)
console = Console()


def _find_latest_replay(cache: Path) -> Optional[Path]:
    paths = sorted(cache.glob("replay_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0] if paths else None


@app.command()
def main(
    start_date: Optional[str] = typer.Option(None, "--start-date"),
    end_date: Optional[str] = typer.Option(None, "--end-date"),
    replay: Optional[Path] = typer.Option(None, "--replay", help="Explicit replay parquet path"),
    mock_llm: bool = typer.Option(False, "--mock-llm", help="Deterministic agents (no OpenAI)"),
    model: Optional[str] = typer.Option(None, "--model", help="OpenAI model when not using mock"),
) -> None:
    load_dotenv()
    s = get_settings()
    cache = s.resolved_cache_dir()

    rpath: Optional[Path] = replay
    if rpath is None:
        if start_date and end_date:
            rpath = cache / f"replay_{start_date}_{end_date}.parquet"
        else:
            rpath = _find_latest_replay(cache)
    if rpath is None or not rpath.exists():
        console.print(
            "[red]No replay dataset found.[/red] Run build_dataset or pass --replay / --start-date & --end-date."
        )
        raise typer.Exit(code=1)

    use_mock = bool(mock_llm) or not (s.openai_api_key)

    graph = build_graph(use_mock=use_mock, model=model, settings=s)
    engine = BacktestEngine(rpath, graph, settings=s)
    reports, summary = engine.run()

    table = Table(title="Backtest summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for k, v in summary.model_dump().items():
        table.add_row(k, f"{v}" if v is not None else "")
    console.print(table)
    console.print(f"[green]Reports[/green] {reports}")


if __name__ == "__main__":
    app()
