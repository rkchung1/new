"""Public Polymarket Gamma + CLOB HTTP clients (no auth)."""

from __future__ import annotations

import time
from typing import Any, Optional

import requests

from btc5m_agents.config import Settings, get_settings


class PolymarketClient:
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

    def list_events_by_slugs(
        self,
        slugs: list[str],
        *,
        closed: Optional[bool] = None,
    ) -> list[dict[str, Any]]:
        """GET /events with slug[] query (Gamma)."""
        if not slugs:
            return []
        base = self._s.gamma_base_url.rstrip("/") + "/events"
        # Gamma accepts repeated slug params
        params: list[tuple[str, str]] = [("slug", s) for s in slugs]
        if closed is not None:
            params.append(("closed", str(closed).lower()))
        r = self._request("GET", base, params=params)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            return []
        return data

    def get_prices_history(
        self,
        token_id: str,
        start_ts: int,
        end_ts: int,
        *,
        fidelity: int = 1,
        interval: str = "max",
    ) -> list[dict[str, Any]]:
        """
        CLOB /prices-history for a token (YES asset id).

        See Polymarket docs: market param is the token id string.
        """
        url = self._s.clob_base_url.rstrip("/") + "/prices-history"
        params = {
            "market": token_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "fidelity": fidelity,
            "interval": interval,
        }
        r = self._request("GET", url, params=params)
        r.raise_for_status()
        body = r.json()
        hist = body.get("history") if isinstance(body, dict) else None
        if not isinstance(hist, list):
            return []
        return hist
