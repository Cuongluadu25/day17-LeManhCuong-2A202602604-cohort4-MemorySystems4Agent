from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


# Default model per provider so the lab runs offline (and live) without the
# user having to specify every knob explicitly.
_PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-sonnet-5-5",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o-mini",
}

# Which env var holds the API key for each provider.
_PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "custom": "CUSTOM_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
    "openrouter": "OPENROUTER_API_KEY",
}

# Which env var holds the base URL for providers that need one.
_PROVIDER_BASE_ENV = {
    "custom": "CUSTOM_BASE_URL",
    "ollama": "OLLAMA_BASE_URL",
}


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    - `base_dir` / `data_dir` / `state_dir`: repo paths.
    - `compact_threshold_tokens`: prompt size (in estimated tokens) that
      triggers compaction.
    - `compact_keep_messages`: how many recent messages to keep in full.
    - `model` / `judge_model`: provider settings for the main model and the
      (optional) judge model.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value else default


def _build_provider(prefix: str) -> ProviderConfig:
    """Build a ProviderConfig for a given env prefix.

    `prefix="LLM_"` reads `LLM_PROVIDER`/`LLM_MODEL`/`LLM_TEMPERATURE`;
    `prefix="JUDGE_"` reads `JUDGE_*` and falls back to the `LLM_*` values.
    """

    provider = normalize_provider(
        _env(f"{prefix}PROVIDER") or _env("LLM_PROVIDER") or "openai"
    )
    model_name = (
        _env(f"{prefix}MODEL")
        or _env("LLM_MODEL")
        or _PROVIDER_DEFAULT_MODELS.get(provider, "gpt-4o-mini")
    )
    temperature = float(
        _env(f"{prefix}TEMPERATURE") or _env("LLM_TEMPERATURE") or "0.0"
    )

    key_env = _PROVIDER_KEY_ENV.get(provider)
    base_env = _PROVIDER_BASE_ENV.get(provider)

    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=_env(key_env) if key_env else None,
        base_url=_env(base_env) if base_env else None,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Resolve the repo root, load `.env` (if present), and return a LabConfig.

    Environment knobs (all optional):
    - LLM_PROVIDER / LLM_MODEL / LLM_TEMPERATURE
    - JUDGE_PROVIDER / JUDGE_MODEL / JUDGE_TEMPERATURE
    - OPENAI_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY
    - OPENROUTER_API_KEY / CUSTOM_BASE_URL / CUSTOM_API_KEY / OLLAMA_BASE_URL
    - COMPACT_THRESHOLD_TOKENS / COMPACT_KEEP_MESSAGES
    """

    root = (Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent).resolve()

    # Load `.env` if python-dotenv is available (never required for offline).
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except Exception:
        pass

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(_env("COMPACT_THRESHOLD_TOKENS", "600")),
        compact_keep_messages=int(_env("COMPACT_KEEP_MESSAGES", "4")),
        model=_build_provider("LLM_"),
        judge_model=_build_provider("JUDGE_"),
    )
