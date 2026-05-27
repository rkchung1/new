"""Label markets and build stratified replay baskets for prompt autoresearch."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import json

from btc5m_agents.config import Settings, get_settings
from btc5m_agents.data.cache import write_parquet


def _subset_slugs(df: pd.DataFrame, slugs: list[str]) -> pd.DataFrame:
    picked = [str(s) for s in slugs]
    missing = set(picked) - set(df["slug"].astype(str).unique())
    if missing:
        raise ValueError(f"slug(s) not in source: {sorted(missing)}")
    sub = df[df["slug"].astype(str).isin(picked)].copy()
    return sub.sort_values(["start_time", "slug", "elapsed"]).reset_index(drop=True)

MIN_ELAPSED_DEFAULT = 101
LATE_ELAPSED_MIN = 240
EARLY_ELAPSED_MAX = 180


@dataclass(frozen=True)
class MarketStats:
    slug: str
    winner: str
    max_abs_gap: float
    gap_range: float
    yes_mid_std: float
    yes_mid_mean: float
    mean_spread: float
    late_gap_delta: float
    abs_late_gap_delta: float
    n_rows: int


@dataclass(frozen=True)
class Thresholds:
    max_abs_gap_p25: float
    max_abs_gap_p90: float
    yes_mid_std_p25: float
    mean_spread_p90: float
    abs_late_gap_delta_p90: float


@dataclass(frozen=True)
class BasketSpec:
    name: str
    kind: str
    description: str
    filename: str


def basket_eligible(spec: BasketSpec, m: MarketStats, t: Thresholds) -> bool:
    k = spec.kind
    if k == "yes_settled":
        return m.winner == "up"
    if k == "no_settled":
        return m.winner == "down"
    if k == "high_btc_move":
        return m.max_abs_gap >= t.max_abs_gap_p90
    if k == "chop":
        return m.yes_mid_std <= t.yes_mid_std_p25 and m.max_abs_gap <= t.max_abs_gap_p25
    if k == "late_reversal":
        return m.abs_late_gap_delta >= t.abs_late_gap_delta_p90
    if k == "wide_spread":
        return m.mean_spread >= t.mean_spread_p90
    raise ValueError(f"Unknown basket kind: {k}")


def basket_rank_key(spec: BasketSpec, m: MarketStats) -> float:
    k = spec.kind
    if k in ("yes_settled", "no_settled", "high_btc_move"):
        return m.max_abs_gap
    if k == "chop":
        return -(m.yes_mid_std + m.max_abs_gap / 1000.0)
    if k == "late_reversal":
        return m.abs_late_gap_delta
    if k == "wide_spread":
        return m.mean_spread
    raise ValueError(f"Unknown basket kind: {k}")


def default_basket_specs() -> list[BasketSpec]:
    return [
        BasketSpec(
            name="yes_settled",
            kind="yes_settled",
            description="Markets that settled UP (YES won); ranked by large |btc_gap|.",
            filename="basket_yes_settled.parquet",
        ),
        BasketSpec(
            name="no_settled",
            kind="no_settled",
            description="Markets that settled DOWN (NO won); ranked by large |btc_gap|.",
            filename="basket_no_settled.parquet",
        ),
        BasketSpec(
            name="high_btc_move",
            kind="high_btc_move",
            description=f"|btc_gap| high (>= p90) over elapsed>={MIN_ELAPSED_DEFAULT}.",
            filename="basket_high_btc_move.parquet",
        ),
        BasketSpec(
            name="chop",
            kind="chop",
            description=f"Tight YES mid (yes_mid_std <= p25) and small BTC move (max |gap| <= p25).",
            filename="basket_chop.parquet",
        ),
        BasketSpec(
            name="late_reversal",
            kind="late_reversal",
            description=f"Large |btc_gap| change early (elapsed<{EARLY_ELAPSED_MAX}) vs late (>={LATE_ELAPSED_MIN}).",
            filename="basket_late_reversal.parquet",
        ),
        BasketSpec(
            name="wide_spread",
            kind="wide_spread",
            description="Wide YES book (mean ask-bid spread >= p90).",
            filename="basket_wide_spread.parquet",
        ),
    ]


BASKET_ASSIGNMENT_ORDER = [
    "late_reversal",
    "wide_spread",
    "high_btc_move",
    "chop",
    "yes_settled",
    "no_settled",
]


def _normalize_winner(winner: object) -> str:
    if winner is None or (isinstance(winner, float) and pd.isna(winner)):
        return ""
    w = str(winner).strip().lower()
    if w in ("up", "yes"):
        return "up"
    if w in ("down", "no"):
        return "down"
    return w


def compute_market_stats(df: pd.DataFrame, *, min_elapsed: int = MIN_ELAPSED_DEFAULT) -> list[MarketStats]:
    """Per-market features for stratification (trading window only)."""
    work = df[df["elapsed"] >= min_elapsed].copy()
    if work.empty:
        return []
    work["yes_mid"] = (work["ask_YES"].astype(float) + work["bid_YES"].astype(float)) / 2.0
    work["spread_yes"] = work["ask_YES"].astype(float) - work["bid_YES"].astype(float)

    out: list[MarketStats] = []
    for slug, g in work.groupby("slug", sort=False):
        g = g.sort_values("elapsed")
        winner = _normalize_winner(g["winner"].iloc[0])
        gap = g["btc_gap"].astype(float)
        yes_mid = g["yes_mid"].astype(float)
        spread = g["spread_yes"].astype(float)
        late = g[g["elapsed"] >= LATE_ELAPSED_MIN]
        early = g[g["elapsed"] < EARLY_ELAPSED_MAX]
        if len(late) and len(early):
            late_delta = float(late["btc_gap"].iloc[-1] - early["btc_gap"].iloc[-1])
        else:
            late_delta = 0.0

        out.append(
            MarketStats(
                slug=str(slug),
                winner=winner,
                max_abs_gap=float(gap.abs().max()),
                gap_range=float(gap.max() - gap.min()),
                yes_mid_std=float(yes_mid.std()) if len(yes_mid) > 1 else 0.0,
                yes_mid_mean=float(yes_mid.mean()),
                mean_spread=float(spread.mean()),
                late_gap_delta=late_delta,
                abs_late_gap_delta=abs(late_delta),
                n_rows=int(len(g)),
            ),
        )
    return out


def compute_thresholds(stats: list[MarketStats]) -> Thresholds:
    if not stats:
        raise ValueError("No market stats to compute thresholds from")
    gaps = np.array([m.max_abs_gap for m in stats])
    stds = np.array([m.yes_mid_std for m in stats])
    spreads = np.array([m.mean_spread for m in stats])
    late = np.array([m.abs_late_gap_delta for m in stats])
    return Thresholds(
        max_abs_gap_p25=float(np.quantile(gaps, 0.25)),
        max_abs_gap_p90=float(np.quantile(gaps, 0.90)),
        yes_mid_std_p25=float(np.quantile(stds, 0.25)),
        mean_spread_p90=float(np.quantile(spreads, 0.90)),
        abs_late_gap_delta_p90=float(np.quantile(late, 0.90)),
    )


def _spec_by_name(specs: list[BasketSpec]) -> dict[str, BasketSpec]:
    return {s.name: s for s in specs}


def assign_baskets(
    stats: list[MarketStats],
    thresholds: Thresholds,
    specs: list[BasketSpec],
    *,
    markets_per_basket: int,
) -> dict[str, list[str]]:
    """Greedy disjoint assignment: specialized baskets first, then settlement."""
    by_name = _spec_by_name(specs)
    order = [by_name[n] for n in BASKET_ASSIGNMENT_ORDER if n in by_name]
    used: set[str] = set()
    assignments: dict[str, list[str]] = {s.name: [] for s in specs}

    for spec in order:
        pool = [m for m in stats if m.slug not in used and basket_eligible(spec, m, thresholds)]
        pool.sort(key=lambda m: basket_rank_key(spec, m), reverse=True)
        picked = [m.slug for m in pool[:markets_per_basket]]
        assignments[spec.name] = picked
        used.update(picked)

    return assignments


def validate_slug_in_basket(
    m: MarketStats,
    spec: BasketSpec,
    thresholds: Thresholds,
) -> tuple[bool, str]:
    if not basket_eligible(spec, m, thresholds):
        return False, "does not meet basket eligibility thresholds"
    return True, "ok"


def validate_assignments(
    stats: list[MarketStats],
    assignments: dict[str, list[str]],
    specs: list[BasketSpec],
    thresholds: Thresholds,
) -> dict[str, Any]:
    by_slug = {m.slug: m for m in stats}
    by_name = _spec_by_name(specs)
    report: dict[str, Any] = {"baskets": {}, "all_passed": True}

    for name, slugs in assignments.items():
        spec = by_name[name]
        rows = []
        for slug in slugs:
            m = by_slug.get(slug)
            if m is None:
                rows.append({"slug": slug, "passed": False, "reason": "slug not in source"})
                report["all_passed"] = False
                continue
            ok, reason = validate_slug_in_basket(m, spec, thresholds)
            rows.append({"slug": slug, "passed": ok, "reason": reason, **asdict(m)})
            if not ok:
                report["all_passed"] = False
        passed = sum(1 for r in rows if r.get("passed"))
        report["baskets"][name] = {
            "description": spec.description,
            "passed": passed,
            "total": len(slugs),
            "slugs": slugs,
            "markets": rows,
        }
    return report


def _rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def basket_separation_summary(
    stats: list[MarketStats],
    assignments: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    """Mean metrics per basket vs full universe (for sanity-checking labels)."""
    by_slug = {m.slug: m for m in stats}
    universe = {
        "max_abs_gap": float(np.mean([m.max_abs_gap for m in stats])),
        "yes_mid_std": float(np.mean([m.yes_mid_std for m in stats])),
        "mean_spread": float(np.mean([m.mean_spread for m in stats])),
        "abs_late_gap_delta": float(np.mean([m.abs_late_gap_delta for m in stats])),
    }
    out: dict[str, dict[str, float]] = {"_universe_mean": universe}
    for name, slugs in assignments.items():
        ms = [by_slug[s] for s in slugs if s in by_slug]
        if not ms:
            continue
        out[name] = {
            "max_abs_gap": float(np.mean([m.max_abs_gap for m in ms])),
            "yes_mid_std": float(np.mean([m.yes_mid_std for m in ms])),
            "mean_spread": float(np.mean([m.mean_spread for m in ms])),
            "abs_late_gap_delta": float(np.mean([m.abs_late_gap_delta for m in ms])),
        }
    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_stratified_baskets(
    source: Path,
    *,
    out_dir: Path,
    manifest_path: Path,
    markets_per_basket: int = 5,
    min_elapsed: int = MIN_ELAPSED_DEFAULT,
    specs: Optional[list[BasketSpec]] = None,
    settings: Optional[Settings] = None,
) -> dict[str, Any]:
    """Build basket parquets + manifest; returns validation report."""
    s = settings or get_settings()
    if not source.is_absolute():
        source = s.project_root / source
    if not source.exists():
        raise FileNotFoundError(f"Source replay not found: {source}")

    raw = pd.read_parquet(source)
    stats = compute_market_stats(raw, min_elapsed=min_elapsed)
    if not stats:
        raise ValueError("No markets after min_elapsed filter")

    thresholds = compute_thresholds(stats)
    basket_specs = specs or default_basket_specs()
    assignments = assign_baskets(
        stats,
        thresholds,
        basket_specs,
        markets_per_basket=markets_per_basket,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    basket_entries = []
    for spec in basket_specs:
        slugs = assignments.get(spec.name, [])
        out_path = out_dir / spec.filename
        if slugs:
            sub = _subset_slugs(raw, slugs)
            write_parquet(sub, out_path)
        else:
            out_path = None

        basket_entries.append(
            {
                "name": spec.name,
                "description": spec.description,
                "file": _rel_path(out_path, s.project_root) if out_path else None,
                "slugs": slugs,
                "markets": len(slugs),
            },
        )

    validation = validate_assignments(stats, assignments, basket_specs, thresholds)
    separation = basket_separation_summary(stats, assignments)

    try:
        source_rel = str(source.relative_to(s.project_root))
    except ValueError:
        source_rel = str(source)

    manifest: dict[str, Any] = {
        "source": source_rel,
        "source_sha256": _sha256_file(source),
        "min_elapsed": min_elapsed,
        "markets_per_basket": markets_per_basket,
        "thresholds": asdict(thresholds),
        "assignment_order": BASKET_ASSIGNMENT_ORDER,
        "baskets": basket_entries,
        "validation": validation,
        "separation_summary": separation,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    stats_csv = manifest_path.parent / "market_stats.csv"
    pd.DataFrame([asdict(m) for m in stats]).to_csv(stats_csv, index=False)

    return manifest


def load_manifest(path: Path, settings: Optional[Settings] = None) -> dict[str, Any]:
    s = settings or get_settings()
    p = path if path.is_absolute() else s.project_root / path
    return json.loads(p.read_text(encoding="utf-8"))


def revalidate_manifest(
    manifest_path: Path,
    settings: Optional[Settings] = None,
) -> dict[str, Any]:
    """Re-score slugs in an existing manifest against criteria (no rebuild)."""
    s = settings or get_settings()
    manifest = load_manifest(manifest_path, s)
    src = Path(manifest["source"])
    if not src.is_absolute():
        src = s.project_root / src
    stats = compute_market_stats(pd.read_parquet(src), min_elapsed=int(manifest.get("min_elapsed", MIN_ELAPSED_DEFAULT)))
    thresholds = Thresholds(**manifest["thresholds"])
    specs = _spec_by_name(default_basket_specs())
    assignments = {b["name"]: b["slugs"] for b in manifest["baskets"]}
    order_names = [n for n in BASKET_ASSIGNMENT_ORDER if n in specs]
    ordered_specs = [specs[n] for n in order_names]
    return validate_assignments(stats, assignments, ordered_specs, thresholds)
