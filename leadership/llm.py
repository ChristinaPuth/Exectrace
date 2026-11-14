"""One LLM entry point for OpenAI and Groq.

Every call walks the provider chain from config.toml (Groq first by default):

1. Pace the call so the provider stays under its tokens-per-minute limit.
2. If the wait would be longer than limits.max_throttle_wait_seconds, or the
   provider is rate-limited or unreachable, move to the next provider.
3. The last provider in the chain is never skipped: it waits as long as needed
   and its errors are raised.

Response caching is added in M3.
"""

from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import openai
from openai import OpenAI

from .config import PROVIDERS, Config, ConfigError, resolve_chain

# Errors that mean "this provider cannot serve the call right now".
# Authentication and bad-request errors are not here: they are configuration
# or code problems and must surface, not be hidden by a fallback.
_UNAVAILABLE = (openai.RateLimitError, openai.APIConnectionError,
                openai.APITimeoutError, openai.InternalServerError)

_OUTPUT_RESERVE = 800   # tokens assumed for the reply when estimating a call


def make_client(provider: str, cfg: Config, max_retries: int | None = None, key: str | None = None) -> OpenAI:
    """A client for one provider. `key`, when given, is used instead of the environment's (a web visitor's own)."""
    p = PROVIDERS[provider]
    key = (os.environ.get(p["key_env"], "") if key is None else key).strip()
    if not key:
        raise ConfigError(f"{p['key_env']} is not set in .env.")
    # The SDK retries timeouts, 429s and 5xx with backoff and honors Retry-After.
    return OpenAI(
        api_key=key,
        base_url=p["base_url"],
        timeout=cfg.limits.request_timeout_seconds,
        max_retries=cfg.limits.max_retries if max_retries is None else max_retries,
    )


def check_key(cfg: Config, provider: str, key: str) -> tuple[bool, str]:
    """Whether a key works for this provider's chat model: lists the provider's models, which uses no tokens.

    The message never contains the key or the provider's error text (which can echo part of it)."""
    if provider not in PROVIDERS:
        return False, f"Unknown provider {provider!r}."
    if not key.strip():
        return False, "Enter a key first."
    model = getattr(cfg.models, provider).chat
    try:
        available = {m.id for m in make_client(provider, cfg, max_retries=0, key=key).models.list()}
    except openai.AuthenticationError:
        return False, f"{provider.capitalize()} rejected this key."
    except openai.OpenAIError as e:
        return False, f"Could not check the key with {provider.capitalize()} ({type(e).__name__}). Try again."
    if model not in available:
        return False, f"The key works, but it cannot use the chat model {model}."
    return True, f"The key works with {model}."


def redact(text: str) -> str:
    """Remove configured secrets, and the SEC contact address, from text before it is saved."""
    for name in ("OPENAI_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY", "SEC_USER_AGENT",
                 "WIKIPEDIA_USER_AGENT"):
        value = os.environ.get(name, "").strip()
        if value:
            text = text.replace(value, "***")
    return text


def estimate_tokens(messages: list[dict]) -> int:
    """Rough, deliberately high estimate: about 3 characters per token."""
    chars = sum(len(str(m.get("content") or "")) for m in messages)
    return chars // 3 + _OUTPUT_RESERVE


class Throttle:
    """Tracks tokens used in the last 60 seconds for one provider."""

    WINDOW = 60.0

    def __init__(self, tokens_per_minute: int, clock: Callable[[], float] = time.monotonic):
        self.limit = tokens_per_minute
        self._clock = clock
        self._used: deque[tuple[float, int]] = deque()

    def wait_needed(self, tokens: int) -> float | None:
        """Seconds to wait before a call of this size fits. 0 = go now.

        None means it can never fit: the call alone exceeds the limit.
        """
        if not self.limit:
            return 0.0
        if tokens > self.limit:
            return None
        now = self._clock()
        while self._used and now - self._used[0][0] >= self.WINDOW:
            self._used.popleft()
        excess = sum(t for _, t in self._used) + tokens - self.limit
        if excess <= 0:
            return 0.0
        for ts, t in self._used:
            excess -= t
            if excess <= 0:
                return max(0.0, ts + self.WINDOW - now)
        return self.WINDOW

    def record(self, tokens: int) -> None:
        self._used.append((self._clock(), tokens))


def decide(wait: float | None, max_wait: float, is_last: bool) -> str:
    """'go' (after waiting `wait` seconds) or 'skip' to the next provider."""
    if is_last:
        return "go"
    if wait is None or wait > max_wait:
        return "skip"
    return "go"


@dataclass
class Completion:
    response: object            # the SDK's ChatCompletion
    provider: str
    model: str
    waited_s: float = 0.0
    skipped: list[str] = field(default_factory=list)   # why earlier providers were passed over


class LLM:
    """Chat completions for one role ('chat' or 'extract') over the provider chain."""

    def __init__(self, cfg: Config, role: str, *,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 keys: Mapping[str, str] | None = None):
        """`keys` ({provider: key}), when given, replaces the environment's keys for this object only.
        The web page passes a visitor's own key this way; it is never put in os.environ, which every
        visitor of the same server shares."""
        self.cfg = cfg
        self._keys = dict(keys) if keys is not None else None
        env = None
        if self._keys is not None:
            key_vars = {p["key_env"] for p in PROVIDERS.values()}
            env = {k: v for k, v in os.environ.items() if k not in key_vars}
            env |= {PROVIDERS[p]["key_env"]: v for p, v in self._keys.items()}
        self.chain = resolve_chain(cfg, role, env)
        self._sleep = sleep
        self._throttles = {p: Throttle(getattr(cfg.models, p).tokens_per_minute, clock)
                           for p, _ in self.chain}
        self._clients: dict[str, OpenAI] = {}

    def _client(self, provider: str, is_last: bool) -> OpenAI:
        if provider not in self._clients:
            # Earlier providers fail fast so the fallback is prompt; the last one retries fully.
            self._clients[provider] = make_client(
                provider, self.cfg, max_retries=None if is_last else 1,
                key=self._keys.get(provider, "") if self._keys is not None else None)
        return self._clients[provider]

    def chat(self, messages: list[dict], **kwargs) -> Completion:
        estimate = estimate_tokens(messages)
        max_wait = self.cfg.limits.max_throttle_wait_seconds
        skipped: list[str] = []
        for i, (provider, model) in enumerate(self.chain):
            is_last = i == len(self.chain) - 1
            throttle = self._throttles[provider]
            wait = throttle.wait_needed(estimate)
            if decide(wait, max_wait, is_last) == "skip":
                skipped.append(f"{provider}: call exceeds its tokens-per-minute limit" if wait is None
                               else f"{provider}: pacing wait {wait:.0f}s is over the {max_wait:.0f}s limit")
                continue
            if wait:
                self._sleep(wait)
            try:
                response = self._client(provider, is_last).chat.completions.create(
                    model=model, messages=messages, **kwargs)
            except _UNAVAILABLE as e:
                if is_last:
                    raise
                skipped.append(f"{provider}: {type(e).__name__}")
                continue
            used = response.usage.total_tokens if response.usage else estimate
            throttle.record(used)
            return Completion(response, provider, model, wait or 0.0, skipped)
        raise AssertionError("unreachable: the last provider is never skipped")
