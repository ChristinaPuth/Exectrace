"""Settings: secrets from .env, everything else from config.toml."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.toml"

# Both providers speak the OpenAI API; only the key and base URL differ.
PROVIDERS: dict[str, dict[str, str | None]] = {
    "openai": {"key_env": "OPENAI_API_KEY", "base_url": None},
    "groq": {"key_env": "GROQ_API_KEY", "base_url": "https://api.groq.com/openai/v1"},
}


class ConfigError(Exception):
    """A setting is missing or invalid. The message is safe to show the user."""


class _Strict(BaseModel):
    # Unknown keys are errors, so a typo in config.toml cannot be silently ignored.
    model_config = ConfigDict(extra="forbid")


class ProviderModels(_Strict):
    chat: str = ""
    extract: str = ""
    tokens_per_minute: int = Field(0, ge=0)   # 0 = do not throttle
    requests_per_day: int = Field(0, ge=0)


class Models(_Strict):
    provider_order: list[str] = ["groq", "openai"]
    openai: ProviderModels = ProviderModels()
    groq: ProviderModels = ProviderModels()


class Limits(_Strict):
    max_pages: int = Field(gt=0)
    max_chunks_per_page: int = Field(gt=0)
    max_search_rounds: int = Field(ge=0)
    max_search_queries: int = Field(ge=0)
    max_llm_calls: int = Field(gt=0)
    max_verification_checks: int = Field(ge=0)
    max_queries_per_verification: int = Field(ge=0)
    max_enrichment_people: int = Field(ge=0)
    max_retries: int = Field(ge=0)
    max_throttle_wait_seconds: float = Field(ge=0)
    max_source_attempts: int = Field(gt=0)
    host_delay_seconds: float = Field(ge=0)
    request_timeout_seconds: float = Field(gt=0)
    freshness_window_months: int = Field(gt=0)


class Discovery(_Strict):
    site_pages: int = Field(12, gt=0)                 # company pages the link picker may choose
    link_candidates: int = Field(120, gt=0)           # links offered to the link picker
    max_sitemaps: int = Field(4, ge=0)                # sitemap files read per company
    search_results_per_query: int = Field(5, gt=0)
    news_pages: int = Field(10, ge=0)                 # search results fetched as pages
    filings_8k: int = Field(3, ge=0)                  # recent 8-K filings on officer changes


class Agent(_Strict):
    max_steps: int = Field(gt=0)
    max_sql_retries: int = Field(ge=0)
    sql_row_limit: int = Field(gt=0)
    sql_timeout_seconds: float = Field(gt=0)


class Eval(_Strict):
    min_precision_current_roles: float = Field(ge=0, le=1)
    min_recall_gold_c_level: float = Field(ge=0, le=1)


class Config(_Strict):
    models: Models
    limits: Limits
    discovery: Discovery = Discovery()
    agent: Agent
    eval: Eval


def load_config(path: Path = CONFIG_PATH) -> Config:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {path}") from None
    try:
        cfg = Config.model_validate(raw)
    except ValueError as e:
        raise ConfigError(f"Invalid {path.name}:\n{e}") from None
    unknown = [p for p in cfg.models.provider_order if p not in PROVIDERS]
    if unknown or not cfg.models.provider_order:
        raise ConfigError(f"Invalid provider_order in {path.name}. Use: {', '.join(PROVIDERS)}.")
    return cfg


def load_env() -> None:
    """Load .env from the project root. Real environment variables win."""
    load_dotenv(ROOT / ".env")


def configured_providers(env: Mapping[str, str] | None = None) -> list[str]:
    """Providers that have a non-empty key, in a stable order."""
    env = os.environ if env is None else env
    return [name for name, p in PROVIDERS.items() if env.get(p["key_env"], "").strip()]


def resolve_chain(cfg: Config, role: str, env: Mapping[str, str] | None = None) -> list[tuple[str, str]]:
    """Ordered (provider, model) pairs to try for role 'chat' or 'extract'.

    Order comes from config.toml provider_order. <ROLE>_PROVIDER in .env moves
    one provider to the front, and <ROLE>_MODEL overrides that provider's
    model. Providers without a key or without a model are left out.
    """
    env = os.environ if env is None else env
    available = configured_providers(env)
    if not available:
        raise ConfigError("No LLM key found. Set OPENAI_API_KEY or GROQ_API_KEY in .env.")

    order = list(cfg.models.provider_order)
    preferred = env.get(f"{role.upper()}_PROVIDER")
    if preferred:
        if preferred not in PROVIDERS:
            raise ConfigError(f"Unknown provider '{preferred}'. Use one of: {', '.join(PROVIDERS)}.")
        order = [preferred] + [p for p in order if p != preferred]

    model_override = env.get(f"{role.upper()}_MODEL")
    chain = []
    for i, provider in enumerate(p for p in order if p in available):
        model = (model_override if i == 0 and model_override else None) \
            or getattr(getattr(cfg.models, provider), role)
        if model:
            chain.append((provider, model))
    if not chain:
        raise ConfigError(
            f"No {role} model set for the providers that have keys ({', '.join(available)}). "
            f"Run `leadership check`, then set [models.<provider>] {role} in config.toml."
        )
    return chain


def resolve_model(cfg: Config, role: str, env: Mapping[str, str] | None = None) -> tuple[str, str]:
    """The first (provider, model) that will be tried for a role."""
    return resolve_chain(cfg, role, env)[0]
