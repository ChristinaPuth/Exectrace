import pytest

from leadership.check import list_chat_models
from leadership.config import (CONFIG_PATH, ConfigError, configured_providers,
                               load_config, resolve_chain, resolve_model)
from leadership.llm import redact

BOTH = {"OPENAI_API_KEY": "a", "GROQ_API_KEY": "b"}


def _write(tmp_path, old, new):
    text = CONFIG_PATH.read_text(encoding="utf-8")
    assert old in text
    bad = tmp_path / "config.toml"
    bad.write_text(text.replace(old, new), encoding="utf-8")
    return bad


def test_committed_config_is_valid():
    cfg = load_config()
    assert cfg.limits.max_pages > 0
    assert cfg.limits.freshness_window_months == 12
    assert cfg.eval.min_recall_gold_c_level == 1.0
    assert cfg.models.provider_order == ["groq", "openai"]


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, "max_pages", "max_pagez"))


def test_invalid_value_is_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, "max_pages = 40", "max_pages = 0"))


def test_unknown_provider_in_order_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="provider_order"):
        load_config(_write(tmp_path, '["groq", "openai"]', '["groq", "acme"]'))


def test_blank_keys_do_not_count_as_configured():
    assert configured_providers({"OPENAI_API_KEY": "  ", "GROQ_API_KEY": "k"}) == ["groq"]
    assert configured_providers({}) == []


def _cfg():
    # Set every model explicitly so the tests do not depend on config.toml's choices.
    cfg = load_config()
    cfg.models.provider_order = ["groq", "openai"]
    cfg.models.openai.chat, cfg.models.openai.extract = "oa-chat", "oa-extract"
    cfg.models.groq.chat, cfg.models.groq.extract = "gq-chat", "gq-extract"
    return cfg


def test_chain_follows_provider_order():
    assert resolve_chain(_cfg(), "extract", BOTH) == [("groq", "gq-extract"), ("openai", "oa-extract")]
    assert resolve_model(_cfg(), "chat", BOTH) == ("groq", "gq-chat")


def test_one_key_is_enough():
    assert resolve_chain(_cfg(), "chat", {"OPENAI_API_KEY": "a"}) == [("openai", "oa-chat")]
    assert resolve_chain(_cfg(), "chat", {"GROQ_API_KEY": "b"}) == [("groq", "gq-chat")]


def test_env_moves_a_provider_to_the_front_and_overrides_its_model():
    env = {**BOTH, "CHAT_PROVIDER": "openai", "CHAT_MODEL": "other"}
    assert resolve_chain(_cfg(), "chat", env) == [("openai", "other"), ("groq", "gq-chat")]


def test_provider_without_a_model_is_left_out():
    cfg = _cfg()
    cfg.models.groq.extract = ""
    assert resolve_chain(cfg, "extract", BOTH) == [("openai", "oa-extract")]


def test_no_key_and_no_model_give_clear_errors():
    with pytest.raises(ConfigError, match="No LLM key"):
        resolve_chain(_cfg(), "chat", {})
    cfg = _cfg()
    cfg.models.openai.extract = ""
    with pytest.raises(ConfigError, match="No extract model"):
        resolve_chain(cfg, "extract", {"OPENAI_API_KEY": "a"})


def test_redact_removes_configured_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-123")
    assert "sk-test-secret-123" not in redact("error for key sk-test-secret-123 here")


def test_model_listing_hides_non_chat_models():
    class Fake:
        class models:
            @staticmethod
            def list():
                return [type("M", (), {"id": i}) for i in
                        ("chat-b", "text-embedding-3", "whisper-large", "chat-a")]
    assert list_chat_models(Fake) == ["chat-a", "chat-b"]
