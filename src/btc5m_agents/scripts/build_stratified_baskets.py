"""Build stratified replay baskets + manifest for prompt autoresearch."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from btc5m_agents.config import get_settings
from btc5m_agents.data.stratified_baskets import (
    build_stratified_baskets,
    load_manifest,
    revalidate_manifest,
)

app = typer.Typer(add_completion=False)
console = Console()


def _print_separation(manifest: dict) -> None:
    sep = manifest.get("separation_summary") or {}
    uni = sep.get("_universe_mean", {})
    if not uni:
        return
    table = Table(title="Basket vs universe (mean metrics)")
    table.add_column("Basket")
    table.add_column("|btc_gap|", justify="right")
    table.add_column("yes_std", justify="right")
    table.add_column("spread", justify="right")
    table.add_column("late_Δ", justify="right")
    table.add_row(
        "universe",
        f"{uni.get('max_abs_gap', 0):.1f}",
        f"{uni.get('yes_mid_std', 0):.4f}",
        f"{uni.get('mean_spread', 0):.4f}",
        f"{uni.get('abs_late_gap_delta', 0):.1f}",
    )
    for name, vals in sep.items():
        if name.startswith("_"):
            continue
        table.add_row(
            name,
            f"{vals.get('max_abs_gap', 0):.1f}",
            f"{vals.get('yes_mid_std', 0):.4f}",
            f"{vals.get('mean_spread', 0):.4f}",
            f"{vals.get('abs_late_gap_delta', 0):.1f}",
        )
    console.print(table)


def _print_validation(manifest: dict) -> None:
    validation = manifest.get("validation") or {}
    table = Table(title="Basket validation")
    table.add_column("Basket")
    table.add_column("Pass", justify="right")
    table.add_column("Slugs")

    for name, info in validation.get("baskets", {}).items():
        passed = info.get("passed", 0)
        total = info.get("total", 0)
        slugs = ", ".join(info.get("slugs", [])[:3])
        if total > 3:
            slugs += f" (+{total - 3})"
        ok = "yes" if passed == total and total > 0 else "NO"
        table.add_row(name, f"{passed}/{total} {ok}", slugs)

    console.print(table)
    if validation.get("all_passed"):
        console.print("[green]All basket assignments pass eligibility checks.[/green]")
    else:
        console.print("[red]Some slugs failed validation — review manifest.[/red]")


@app.command()
def main(
    source: Optional[Path] = typer.Option(
        None,
        "--source",
        help="Full btc_5m_2s.parquet (default: data/cache/btc_5m_2s.parquet)",
    ),
    out_dir: Optional[Path] = typer.Option(
        None,
        "--out-dir",
        help="Basket parquet output dir (default: data/cache/baskets)",
    ),
    manifest: Optional[Path] = typer.Option(
        None,
        "--manifest",
        help="Manifest JSON path (default: autoresearch/baskets/manifest.json)",
    ),
    markets_per_basket: int = typer.Option(
        5,
        "--markets-per-basket",
        min=1,
        help="Markets per basket (disjoint across baskets when possible)",
    ),
    validate_only: bool = typer.Option(
        False,
        "--validate-only",
        help="Re-check existing manifest slugs; do not rebuild parquets",
    ),
) -> None:
    s = get_settings()
    src = source or s.btc_5m_replay_path
    if not src.is_absolute():
        src = s.project_root / src

    baskets_dir = out_dir or (s.resolved_cache_dir() / "baskets")
    if not baskets_dir.is_absolute():
        baskets_dir = s.project_root / baskets_dir

    manifest_path = manifest or (s.project_root / "autoresearch" / "baskets" / "manifest.json")
    if not manifest_path.is_absolute():
        manifest_path = s.project_root / manifest_path

    if validate_only:
        if not manifest_path.exists():
            console.print(f"[red]Manifest not found:[/red] {manifest_path}")
            raise typer.Exit(code=1)
        m = load_manifest(manifest_path, s)
        m["validation"] = revalidate_manifest(manifest_path, s)
        _print_validation(m)
        raise typer.Exit(code=0 if m["validation"].get("all_passed") else 1)

    if not src.exists():
        console.print(f"[red]Source not found:[/red] {src}")
        raise typer.Exit(code=1)

    manifest_data = build_stratified_baskets(
        src,
        out_dir=baskets_dir,
        manifest_path=manifest_path,
        markets_per_basket=markets_per_basket,
    )

    console.print(f"[green]Wrote baskets[/green] → {baskets_dir}")
    console.print(f"[green]Manifest[/green] → {manifest_path}")
    console.print(f"[green]Market stats[/green] → {manifest_path.parent / 'market_stats.csv'}")

    for b in manifest_data.get("baskets", []):
        console.print(f"  {b['name']}: {b['markets']} markets → {b['file']}")

    _print_separation(manifest_data)
    _print_validation(manifest_data)
    if not manifest_data.get("validation", {}).get("all_passed"):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
