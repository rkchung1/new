"""Chat model factory for OpenAI and vLLM (OpenAI-compatible) backends."""

from __future__ import annotations

from typing import Literal, Optional

import requests
from langchain_openai import ChatOpenAI

from btc5m_agents.config import Settings, get_settings

LlmBackend = Literal["openai", "vllm"]


def resolve_backend(settings: Settings, override: Optional[str] = None) -> LlmBackend:
    raw = (override or settings.llm_backend).strip().lower()
    if raw not in ("openai", "vllm"):
        raise ValueError(f"Invalid llm_backend '{raw}'; use 'openai' or 'vllm'.")
    return raw  # type: ignore[return-value]


def resolve_model_name(
    settings: Settings,
    *,
    backend: LlmBackend,
    model: Optional[str] = None,
) -> str:
    if model:
        return model
    if backend == "vllm" and settings.vllm_model:
        return settings.vllm_model
    return settings.openai_model


def build_chat_model(
    settings: Optional[Settings] = None,
    *,
    model: Optional[str] = None,
    llm_backend: Optional[str] = None,
    vllm_base_url: Optional[str] = None,
) -> ChatOpenAI:
    """Build a ChatOpenAI client for OpenAI or a local vLLM server."""
    s = settings or get_settings()
    backend = resolve_backend(s, llm_backend)
    model_name = resolve_model_name(s, backend=backend, model=model)
    timeout = s.llm_timeout_sec

    max_tokens = s.llm_max_tokens

    if backend == "vllm":
        base = (vllm_base_url or s.vllm_base_url).rstrip("/")
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=s.vllm_api_key,
            base_url=base,
            timeout=timeout,
            max_tokens=max_tokens,
        )

    if not s.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is required when LLM_BACKEND=openai. "
            "Set the key in .env or use --llm-backend vllm / --mock-llm.",
        )
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=s.openai_api_key,
        timeout=timeout,
        max_tokens=max_tokens,
    )


def check_vllm_reachable(
    settings: Optional[Settings] = None,
    *,
    vllm_base_url: Optional[str] = None,
    timeout_sec: float = 5.0,
) -> None:
    """Verify the vLLM server responds before starting a long backtest."""
    s = settings or get_settings()
    base = (vllm_base_url or s.vllm_base_url).rstrip("/")
    url = f"{base}/models"
    try:
        resp = requests.get(url, timeout=timeout_sec)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ConnectionError(
            f"Cannot reach vLLM at {url}. Start the server, e.g. "
            f"'vllm serve <model> --port 8000', and set VLLM_BASE_URL={base}.",
        ) from exc
