"""Build a small btc_5m_2s parquet slice for fast LLM / graph smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console

from btc5m_agents.config import get_settings
from btc5m_agents.data.cache import write_parquet

app = typer.Typer(add_completion=False)
console = Console()

_DEFAULT_OUT_NAME = "btc_5m_2s_smoke.parquet"
_PRESET_OUT = {
    3: "btc_5m_2s_smoke.parquet",
    60: "btc_5m_2s_60m.parquet",
}


def build_replay_subset(
    source: Path,
    *,
    max_markets: int = 3,
    slugs: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Return raw replay rows for the first N markets (or explicit slugs), in source order."""
    df = pd.read_parquet(source)
    if "slug" not in df.columns:
        raise ValueError("Replay parquet must have a slug column")

    if slugs:
        picked = [str(s) for s in slugs]
        missing = set(picked) - set(df["slug"].astype(str).unique())
        if missing:
            raise ValueError(f"slug(s) not in source: {sorted(missing)}")
    else:
        order = (
            df.groupby("slug")["start_time"]
            .first()
            .sort_values()
            .index.astype(str)
            .tolist()
        )
        n = max(1, max_markets)
        picked = order[:n]

    sub = df[df["slug"].astype(str).isin(picked)].copy()
    sub = sub.sort_values(["start_time", "slug", "elapsed"]).reset_index(drop=True)
    return sub


@app.command()
def main(
    source: Optional[Path] = typer.Option(
        None,
        "--source",
        help="Full btc_5m_2s.parquet (default: data/cache/btc_5m_2s.parquet)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help=f"Output path (default: data/cache/{_DEFAULT_OUT_NAME})",
    ),
    max_markets: int = typer.Option(
        3,
        "--max-markets",
        min=1,
        help="Number of consecutive 5m markets to include (ignored if --slug is set)",
    ),
    slug: Optional[list[str]] = typer.Option(
        None,
        "--slug",
        help="Explicit market slug(s); repeatable. Overrides --max-markets.",
    ),
) -> None:
    s = get_settings()
    src = source or s.btc_5m_replay_path
    if not src.is_absolute():
        src = s.project_root / src
    if not src.exists():
        console.print(f"[red]Source not found:[/red] {src}")
        raise typer.Exit(code=1)

    if output is None and max_markets in _PRESET_OUT and slug is None:
        out = s.resolved_cache_dir() / _PRESET_OUT[max_markets]
    else:
        out = output or (s.resolved_cache_dir() / _DEFAULT_OUT_NAME)
    if not out.is_absolute():
        out = s.project_root / out

    sub = build_replay_subset(src, max_markets=max_markets, slugs=slug)
    markets = sub["slug"].astype(str).nunique()
    write_parquet(sub, out)

    console.print(
        f"[green]Wrote[/green] {out}\n"
        f"  markets: {markets}\n"
        f"  raw rows: {len(sub)}\n"
        f"  slugs: {sub['slug'].astype(str).unique().tolist()}",
    )
    console.print(
        "\n[cyan]Fast LLM backtest example:[/cyan]\n"
        f"  python -m btc5m_agents.scripts.run_backtest --replay {out} "
        "--decision-every-sec 60 --llm-backend vllm",
    )


if __name__ == "__main__":
    app()
