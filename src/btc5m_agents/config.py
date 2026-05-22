"""Application settings (env + defaults)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_project_root() -> Path:
    # src/btc5m_agents/config.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=_default_project_root)

    gamma_base_url: str = "https://gamma-api.polymarket.com"
    clob_base_url: str = "https://clob.polymarket.com"
    coinbase_exchange_url: str = "https://api.exchange.coinbase.com"

    data_dir: Path = Field(default_factory=lambda: _default_project_root() / "data")
    cache_dir: Optional[Path] = None
    reports_dir: Optional[Path] = None

    initial_cash: float = 1000.0
    max_position_per_market_usd: float = 100.0
    max_total_exposure_usd: float = 300.0
    slippage: float = 0.01
    price_floor: float = 0.01
    price_ceiling: float = 0.99

    http_timeout_sec: float = 30.0
    http_max_retries: int = 4
    http_backoff_sec: float = 0.5

    llm_backend: Literal["openai", "vllm"] = Field(default="openai", alias="LLM_BACKEND")
    llm_timeout_sec: float = Field(default=120.0, alias="LLM_TIMEOUT_SEC")
    llm_max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS")

    market_history_sec: int = Field(default=900, alias="MARKET_HISTORY_SEC")

    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    vllm_base_url: str = Field(default="http://localhost:8000/v1", alias="VLLM_BASE_URL")
    vllm_model: Optional[str] = Field(default=None, alias="VLLM_MODEL")
    vllm_api_key: str = Field(default="EMPTY", alias="VLLM_API_KEY")

    slug_prefix: str = "btc-updown-5m"
    snapshot_interval_minutes: int = 1

    btc_5m_replay_path: Path = Field(
        default_factory=lambda: _default_project_root() / "data" / "cache" / "btc_5m_2s.parquet",
    )
    replay_decision_every_sec: int = 10
    btc_5m_data_start_elapsed: int = 101

    def resolved_cache_dir(self) -> Path:
        return self.cache_dir or (self.data_dir / "cache")

    def resolved_reports_dir(self) -> Path:
        return self.reports_dir or (self.project_root / "reports")


def get_settings() -> Settings:
    return Settings()
