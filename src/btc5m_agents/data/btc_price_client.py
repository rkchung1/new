"""Coinbase Exchange public candles (1-minute BTC-USD)."""

from __future__ import annotations

import time
from typing import Any, Optional

import requests

from btc5m_agents.config import Settings, get_settings


class BtcPriceClient:
    """Fetches BTC-USD candles with granularity=60 (1m), chunked to 300 bars per call."""

    GRANULARITY_SEC = 60
    MAX_BARS = 300

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._s = settings or get_settings()
        self._session = requests.Session()

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        timeout = kwargs.pop("timeout", self._s.http_timeout_sec)
        last_exc: Optional[Exception] = None
        for attempt in range(self._s.http_max_retries):
            try:
                r = self._session.request(method, url, timeout=timeout, **kwargs)
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(self._s.http_backoff_sec * (2**attempt))
                    continue
                return r
            except requests.RequestException as e:
                last_exc = e
                time.sleep(self._s.http_backoff_sec * (2**attempt))
        if last_exc:
            raise last_exc
        raise RuntimeError("HTTP request failed without exception")

    def fetch_candles(self, start_ts: int, end_ts: int) -> list[list[Any]]:
        """
        Returns raw Coinbase candle rows: [time, low, high, open, close, volume].
        `time` is bucket start in unix seconds. Replay uses `open` at each bucket start.
        """
        base = self._s.coinbase_exchange_url.rstrip("/") + "/products/BTC-USD/candles"
        window = self.GRANULARITY_SEC * self.MAX_BARS
        all_rows: list[list[Any]] = []
        cursor = start_ts
        while cursor < end_ts:
            chunk_end = min(end_ts, cursor + window)
            params = {
                "granularity": self.GRANULARITY_SEC,
                "start": cursor,
                "end": chunk_end,
            }
            r = self._request("GET", base, params=params)
            r.raise_for_status()
            chunk = r.json()
            if isinstance(chunk, list):
                all_rows.extend(chunk)
            cursor = chunk_end
        # Deduplicate by time
        by_t: dict[int, list[Any]] = {}
        for row in all_rows:
            if isinstance(row, list) and len(row) >= 6:
                by_t[int(row[0])] = row
        return [by_t[k] for k in sorted(by_t.keys())]
