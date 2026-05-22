"""CLI: run LangGraph backtest on btc_5m_2s replay data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from btc5m_agents.agents.graph import build_graph
from btc5m_agents.agents.llm import check_vllm_reachable, resolve_backend, resolve_model_name
from btc5m_agents.backtest.engine import BacktestEngine
from btc5m_agents.config import get_settings

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    start_date: Optional[str] = typer.Option(None, "--start-date", help="Filter markets by start_time (UTC)"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="Filter markets by start_time (UTC)"),
    replay: Optional[Path] = typer.Option(
        None,
        "--replay",
        help="Path to btc_5m_2s.parquet (default: data/cache/btc_5m_2s.parquet)",
    ),
    decision_every_sec: Optional[int] = typer.Option(
        None,
        "--decision-every-sec",
        help="Agent decision interval in seconds (default from settings, usually 10)",
    ),
    mock_llm: bool = typer.Option(False, "--mock-llm", help="Deterministic agents (no LLM)"),
    llm_backend: Optional[str] = typer.Option(
        None,
        "--llm-backend",
        help="LLM backend: openai or vllm (default from LLM_BACKEND env)",
    ),
    vllm_base_url: Optional[str] = typer.Option(
        None,
        "--vllm-base-url",
        help="vLLM OpenAI-compatible base URL (default http://localhost:8000/v1)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Model name (OpenAI model id or vLLM served model name)",
    ),
) -> None:
    load_dotenv()
    s = get_settings()

    rpath = replay or s.btc_5m_replay_path
    if not rpath.is_absolute():
        rpath = s.project_root / rpath
    if not rpath.exists():
        console.print(
            f"[red]Replay not found at {rpath}.[/red] "
            "Place btc_5m_2s.parquet under data/cache/.",
        )
        raise typer.Exit(code=1)

    backend = resolve_backend(s, llm_backend)
    use_mock = bool(mock_llm) or (backend == "openai" and not s.openai_api_key)

    if not use_mock and backend == "vllm":
        check_vllm_reachable(s, vllm_base_url=vllm_base_url)
        model_name = resolve_model_name(s, backend=backend, model=model)
        console.print(
            f"[cyan]LLM[/cyan] vllm @ {vllm_base_url or s.vllm_base_url} model={model_name}",
        )
    elif not use_mock:
        model_name = resolve_model_name(s, backend=backend, model=model)
        console.print(f"[cyan]LLM[/cyan] openai model={model_name}")
    else:
        console.print("[cyan]LLM[/cyan] mock (rule-based agents)")

    graph = build_graph(
        use_mock=use_mock,
        model=model,
        settings=s,
        llm_backend=llm_backend,
        vllm_base_url=vllm_base_url,
    )
    engine = BacktestEngine(
        rpath,
        graph,
        settings=s,
        start_date=start_date,
        end_date=end_date,
        decision_every_sec=decision_every_sec,
        llm_backend=backend if not use_mock else None,
        llm_model=resolve_model_name(s, backend=backend, model=model) if not use_mock else None,
        vllm_base_url=vllm_base_url if backend == "vllm" else None,
    )
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
