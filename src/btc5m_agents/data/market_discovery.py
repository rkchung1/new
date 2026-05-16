"""Discover BTC 5m Up/Down markets via Gamma slug enumeration."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from btc5m_agents.config import Settings, get_settings
from btc5m_agents.data.cache import cache_root, write_parquet
from btc5m_agents.data.polymarket_client import PolymarketClient
from btc5m_agents.types import MarketMeta


def _parse_day(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def _utc_range_ts(start_date: str, end_date: str) -> tuple[int, int]:
    d0 = _parse_day(start_date)
    d1 = _parse_day(end_date)
    start_dt = datetime.combine(d0, time.min, tzinfo=timezone.utc)
    end_dt = datetime.combine(d1, time.max, tzinfo=timezone.utc)
    return int(start_dt.timestamp()), int(end_dt.timestamp())


def five_minute_slug_ts_iter(start_ts: int, end_ts: int) -> list[int]:
    """Unix timestamps aligned to 5-minute boundaries for slug suffix."""
    t0 = (start_ts // 300) * 300
    out: list[int] = []
    t = t0
    while t <= end_ts:
        out.append(t)
        t += 300
    return out


def slug_for_ts(prefix: str, ts: int) -> str:
    return f"{prefix}-{ts}"


def _parse_json_list(s: Optional[str]) -> list[Any]:
    if not s:
        return []
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else []
    except json.JSONDecodeError:
        return []


def settlement_yes_from_market(m: dict[str, Any]) -> Optional[float]:
    """
    Returns 1.0 if YES/Up won, 0.0 if NO/Down won, None if unknown.
    """
    outcomes = _parse_json_list(m.get("outcomes"))
    prices = _parse_json_list(m.get("outcomePrices"))
    if not outcomes or not prices or len(outcomes) != len(prices):
        return None
    try:
        px = [float(x) for x in prices]
    except (TypeError, ValueError):
        return None
    if not px or max(px) < 0.51:
        return None
    idx = int(max(range(len(px)), key=lambda i: px[i]))
    label = str(outcomes[idx]).strip().lower()
    if label in ("up", "yes"):
        return 1.0
    if label in ("down", "no"):
        return 0.0
    if idx == 0:
        return 1.0
    if idx == 1:
        return 0.0
    return None


def event_to_market_meta(event: dict[str, Any], settings: Settings) -> Optional[MarketMeta]:
    _ = settings
    markets = event.get("markets") or []
    if not markets:
        return None
    m0 = markets[0]
    if not isinstance(m0, dict):
        return None
    tokens = _parse_json_list(m0.get("clobTokenIds"))
    if len(tokens) < 2:
        return None
    yes_token = str(tokens[0])
    no_token = str(tokens[1]) if len(tokens) > 1 else None

    start_iso = m0.get("startDate") or event.get("startDate")
    end_iso = m0.get("endDate") or event.get("endDate")
    if not start_iso or not end_iso:
        return None
    try:
        start_ts = int(pd.Timestamp(start_iso).timestamp())
        end_ts = int(pd.Timestamp(end_iso).timestamp())
    except Exception:
        return None

    slug = event.get("slug") or m0.get("slug") or ""
    sett = settlement_yes_from_market(m0)
    return MarketMeta(
        event_slug=str(slug),
        condition_id=str(m0.get("conditionId") or ""),
        market_id=str(m0.get("id") or m0.get("conditionId") or ""),
        yes_token_id=yes_token,
        no_token_id=no_token,
        question=m0.get("question") or event.get("title"),
        start_ts=start_ts,
        end_ts=end_ts,
        outcomes=m0.get("outcomes"),
        outcome_prices=m0.get("outcomePrices"),
        uma_resolution_status=m0.get("umaResolutionStatus"),
        closed=bool(m0.get("closed")),
        settlement_yes=sett,
    )


def discover(
    start_date: str,
    end_date: str,
    *,
    settings: Optional[Settings] = None,
    batch_size: int = 25,
) -> Path:
    """
    Query Gamma for candidate slugs `btc-updown-5m-{unix}` in range; write Parquet bundle.
    Returns path to written markets file.
    """
    s = settings or get_settings()
    client = PolymarketClient(s)
    start_ts, end_ts = _utc_range_ts(start_date, end_date)
    candidates = five_minute_slug_ts_iter(start_ts, end_ts)
    slugs = [slug_for_ts(s.slug_prefix, ts) for ts in candidates]

    events: list[dict[str, Any]] = []
    for i in range(0, len(slugs), batch_size):
        batch = slugs[i : i + batch_size]
        chunk = client.list_events_by_slugs(batch, closed=True)
        if not chunk:
            chunk = client.list_events_by_slugs(batch, closed=False)
        events.extend(chunk)

    metas: list[MarketMeta] = []
    seen: set[str] = set()
    for ev in events:
        if not isinstance(ev, dict):
            continue
        slug = str(ev.get("slug") or "")
        if slug in seen:
            continue
        meta = event_to_market_meta(ev, s)
        if meta is None:
            continue
        if not meta.event_slug.startswith(s.slug_prefix):
            continue
        seen.add(slug)
        metas.append(meta)

    metas.sort(key=lambda m: m.start_ts)
    rows = [m.model_dump() for m in metas]
    df = pd.DataFrame(rows)
    out_dir = cache_root(s) / "markets"
    out_path = out_dir / f"markets_{start_date}_{end_date}.parquet"
    write_parquet(df, out_path)
    return out_path


def load_markets_bundle(
    start_date: str, end_date: str, settings: Optional[Settings] = None
) -> pd.DataFrame:
    s = settings or get_settings()
    path = cache_root(s) / "markets" / f"markets_{start_date}_{end_date}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
