"""Run backtest on every stratified basket listed in manifest.json."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

from btc5m_agents.autoresearch.config import load_autoresearch_config, resolve_path
from btc5m_agents.autoresearch.paths import last_eval_path, prompts_path, prompts_sha256
from btc5m_agents.config import get_settings
from btc5m_agents.data.stratified_baskets import load_manifest

app = typer.Typer(add_completion=False)
console = Console()

_RUN_ID_RE = re.compile(r"^AUTORESEARCH_RUN_ID=(\S+)", re.MULTILINE)


def _parse_run_id(output: str) -> Optional[str]:
    m = _RUN_ID_RE.search(output)
    return m.group(1) if m else None


def _run_one_backtest(
    *,
    replay: Path,
    decision_every_sec: int,
    llm_backend: Optional[str],
    model: Optional[str],
    vllm_base_url: Optional[str],
    cwd: Path,
) -> tuple[int, str, Optional[str]]:
    cmd = [
        sys.executable,
        "-m",
        "btc5m_agents.scripts.run_backtest",
        "--replay",
        str(replay),
        "--decision-every-sec",
        str(decision_every_sec),
    ]
    if llm_backend:
        cmd.extend(["--llm-backend", llm_backend])
    if model:
        cmd.extend(["--model", model])
    if vllm_base_url:
        cmd.extend(["--vllm-base-url", vllm_base_url])

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    run_id = _parse_run_id(combined)
    return proc.returncode, combined, run_id


@app.command()
def main(
    manifest: Optional[Path] = typer.Option(
        None,
        "--manifest",
        help="Basket manifest JSON (default from autoresearch/config.yaml)",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Autoresearch config YAML",
    ),
    decision_every_sec: Optional[int] = typer.Option(None, "--decision-every-sec"),
    llm_backend: Optional[str] = typer.Option(None, "--llm-backend"),
    model: Optional[str] = typer.Option(None, "--model"),
    vllm_base_url: Optional[str] = typer.Option(None, "--vllm-base-url"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands only"),
) -> None:
    load_dotenv()
    s = get_settings()
    cfg_path = resolve_path(config or Path("autoresearch/config.yaml"), s)
    cfg = load_autoresearch_config(cfg_path, s)
    manifest_path = resolve_path(manifest or Path(str(cfg["basket_manifest"])), s)
    every = int(decision_every_sec if decision_every_sec is not None else cfg["decision_every_sec"])

    mdata = load_manifest(manifest_path, s)
    baskets = mdata.get("baskets") or []
    if not baskets:
        console.print("[red]No baskets in manifest.[/red]")
        raise typer.Exit(code=1)

    pp = prompts_path(s)
    if not pp.exists():
        console.print(f"[red]prompts.py not found:[/red] {pp}")
        raise typer.Exit(code=1)

    eval_doc: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "prompts_sha256": prompts_sha256(pp),
        "decision_every_sec": every,
        "manifest": str(manifest_path.relative_to(s.project_root))
        if manifest_path.is_relative_to(s.project_root)
        else str(manifest_path),
        "baskets": {},
    }

    failed = False
    for b in baskets:
        name = str(b["name"])
        rel_file = b.get("file")
        if not rel_file:
            console.print(f"[red]Basket {name} has no file in manifest[/red]")
            failed = True
            continue
        replay = resolve_path(Path(str(rel_file)), s)
        if not replay.exists():
            console.print(f"[red]Replay not found for {name}:[/red] {replay}")
            failed = True
            continue

        if dry_run:
            console.print(f"[cyan]would run[/cyan] {name} → {replay}")
            continue

        console.print(f"[cyan]Backtest[/cyan] {name} …")
        code, output, run_id = _run_one_backtest(
            replay=replay,
            decision_every_sec=every,
            llm_backend=llm_backend,
            model=model,
            vllm_base_url=vllm_base_url,
            cwd=s.project_root,
        )
        if code != 0 or not run_id:
            failed = True
            console.print(f"[red]Failed[/red] {name} (exit={code})")
            eval_doc["baskets"][name] = {
                "replay": str(rel_file),
                "status": "crash",
                "exit_code": code,
                "run_id": run_id,
                "log_tail": output[-2000:] if output else "",
            }
            continue

        reports = s.resolved_reports_dir() / "backtests" / run_id
        eval_doc["baskets"][name] = {
            "replay": str(rel_file),
            "status": "ok",
            "run_id": run_id,
            "reports_dir": str(reports.relative_to(s.project_root))
            if reports.is_relative_to(s.project_root)
            else str(reports),
        }
        console.print(f"  {name} AUTORESEARCH_RUN_ID={run_id}")

    if dry_run:
        return

    out_path = last_eval_path(s)
    out_path.write_text(json.dumps(eval_doc, indent=2), encoding="utf-8")
    console.print(f"[green]Wrote[/green] {out_path}")

    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
