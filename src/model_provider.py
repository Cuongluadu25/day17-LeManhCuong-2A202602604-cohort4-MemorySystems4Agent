from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Shared provider configuration used by both agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


# Aliases -> canonical provider names. Includes common typos so the memory
# system is not locked to a single provider naming convention.
_PROVIDER_ALIASES = {
    "openai": "openai",
    "gpt": "openai",
    "chatgpt": "openai",
    "custom": "custom",
    "openai-compatible": "custom",
    "azure": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "google-genai": "gemini",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "anthorpic": "anthropic",  # common typo
    "ollama": "ollama",
    "openrouter": "openrouter",
}


def normalize_provider(value: str) -> str:
    """Map aliases (e.g. `anthorpic` -> `anthropic`) to canonical names."""
    if not value:
        return "openai"
    key = value.strip().lower()
    return _PROVIDER_ALIASES.get(key, key)


def build_chat_model(config: ProviderConfig):
    """Instantiate the real LangChain chat model for the selected provider.

    Imports are done lazily inside each branch so offline mode (benchmark /
    tests) never requires the provider SDKs to be installed.
    """

    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatOpenAI(**kwargs)

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatAnthropic(**kwargs)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatOpenRouter(**kwargs)

    raise ValueError(f"Unsupported provider: {provider}")
