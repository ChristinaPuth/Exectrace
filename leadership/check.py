"""`leadership check`: prove each configured key works and report what it can use.

Makes real calls. For every provider with a key it lists the models, and for
each model named in config.toml it runs the two capabilities the project
depends on: JSON output (extraction) and tool calling (chat agent).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import openai

from .config import PROVIDERS, Config, configured_providers
from .llm import make_client, redact

# Model families that cannot do chat completions; hidden from the listing.
_NOT_CHAT = ("embed", "whisper", "tts", "dall-e", "image", "audio", "realtime",
             "moderation", "transcribe", "guard", "sora", "speech")

_TOOL = {
    "type": "function",
    "function": {
        "name": "query_db",
        "description": "Run one read-only SQL query and return the rows.",
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        },
    },
}


@dataclass
class Probe:
    model: str
    json_ok: bool = False
    tools_ok: bool = False
    temperature_zero: bool = True
    latency_s: float = 0.0
    tokens: int = 0
    rate_limits: dict[str, str] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.json_ok and self.tools_ok and not self.error


def _describe(e: Exception) -> str:
    if isinstance(e, openai.AuthenticationError):
        # The provider's own message echoes part of the key; do not repeat it.
        return "authentication failed - check the key in .env"
    if isinstance(e, openai.APIStatusError):
        return redact(f"{type(e).__name__} (HTTP {e.status_code}): {str(e)[:300]}")
    return redact(f"{type(e).__name__}: {str(e)[:300]}")


def list_chat_models(client: openai.OpenAI) -> list[str]:
    ids = sorted(m.id for m in client.models.list())
    return [i for i in ids if not any(s in i.lower() for s in _NOT_CHAT)]


def _create(client: openai.OpenAI, probe: Probe, **kwargs):
    """One chat call at temperature 0, falling back if the model forbids it."""
    api = client.chat.completions.with_raw_response
    started = time.perf_counter()
    try:
        raw = api.create(model=probe.model, temperature=0, **kwargs)
    except openai.BadRequestError as e:
        if "temperature" not in str(e).lower():
            raise
        probe.temperature_zero = False
        raw = api.create(model=probe.model, **kwargs)
    probe.latency_s += time.perf_counter() - started
    probe.rate_limits.update(
        {k.lower(): v for k, v in raw.headers.items() if k.lower().startswith("x-ratelimit-")}
    )
    resp = raw.parse()
    if resp.usage:
        probe.tokens += resp.usage.total_tokens
    return resp


def probe_model(client: openai.OpenAI, model: str) -> Probe:
    p = Probe(model=model)
    try:
        resp = _create(
            client, p,
            messages=[
                {"role": "system", "content": "Reply with a JSON object only."},
                {"role": "user", "content": 'Return exactly this JSON: {"ok": true}'},
            ],
            response_format={"type": "json_object"},
        )
        try:
            p.json_ok = json.loads(resp.choices[0].message.content or "").get("ok") is True
        except (ValueError, AttributeError):
            p.json_ok = False

        resp = _create(
            client, p,
            messages=[{"role": "user", "content":
                       "Use the query_db tool to count the rows in the table named people."}],
            tools=[_TOOL],
        )
        calls = resp.choices[0].message.tool_calls or []
        p.tools_ok = bool(calls) and "sql" in json.loads(calls[0].function.arguments)
    except (openai.OpenAIError, ValueError) as e:
        p.error = _describe(e)
    return p


def run(cfg: Config) -> int:
    """Print the report. Exit code: 0 all good, 1 a probe failed, 2 setup needed."""
    env = os.environ
    flags = {name: bool(env.get(p["key_env"], "").strip()) for name, p in PROVIDERS.items()}
    flags["tavily"] = bool(env.get("TAVILY_API_KEY", "").strip())
    flags["sec_user_agent"] = bool(env.get("SEC_USER_AGENT", "").strip())
    print("Keys in .env:  " + "  ".join(f"{k}={'set' if v else 'MISSING'}" for k, v in flags.items()))

    providers = configured_providers()
    if not providers:
        print("\nNo LLM key found. Copy .env.example to .env and set OPENAI_API_KEY or GROQ_API_KEY.")
        return 2

    failed = False
    unset = False
    for name in providers:
        client = make_client(name, cfg)
        print(f"\n[{name}]")
        try:
            models = list_chat_models(client)
        except openai.OpenAIError as e:
            print(f"  could not list models: {_describe(e)}")
            failed = True
            continue
        print(f"  {len(models)} chat-capable models available:")
        for i in range(0, len(models), 3):
            print("    " + "   ".join(models[i:i + 3]))

        pm = getattr(cfg.models, name)
        wanted = sorted({m for m in (pm.chat, pm.extract) if m})
        if not wanted:
            print(f"  no model chosen yet - set [models.{name}] chat / extract in config.toml")
            unset = True
            continue
        for model in wanted:
            p = probe_model(client, model)
            if p.error:
                print(f"  {model}: FAILED - {p.error}")
            else:
                print(f"  {model}: JSON output {'ok' if p.json_ok else 'FAILED'}, "
                      f"tool calling {'ok' if p.tools_ok else 'FAILED'}, "
                      f"temperature 0 {'ok' if p.temperature_zero else 'not supported'}, "
                      f"{p.latency_s:.1f}s, {p.tokens} tokens")
            for k in sorted(p.rate_limits):
                print(f"      {k} = {p.rate_limits[k]}")
            failed = failed or not p.ok

    if failed:
        return 1
    if unset:
        print("\nNext: choose models from the lists above, set them in config.toml, run this again.")
        return 2
    print("\nAll configured models passed.")
    return 0
