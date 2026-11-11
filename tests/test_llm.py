"""Provider chain: Groq first, paced, then OpenAI.

The pacing logic is pure and tested without any model. The fallback paths are
tested against the real providers (marked `llm`); no completion is mocked.
"""

import pytest

from leadership.config import PROVIDERS, load_config, load_env
from leadership.llm import LLM, Throttle, decide, estimate_tokens


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


# --- Throttle (fast) --------------------------------------------------------

def test_no_limit_never_waits():
    assert Throttle(0).wait_needed(10**9) == 0.0


def test_call_that_fits_goes_now():
    t = Throttle(8000, Clock())
    t.record(3000)
    assert t.wait_needed(5000) == 0.0


def test_wait_lasts_until_enough_old_usage_expires():
    clock = Clock()
    t = Throttle(8000, clock)
    t.record(5000)              # at 1000
    clock.now = 1020.0
    t.record(2000)              # at 1020
    clock.now = 1030.0
    # 7000 used + 4000 wanted = 3000 over; the 5000 call expires at 1060.
    assert t.wait_needed(4000) == pytest.approx(30.0)
    # 7000 wanted: both calls must expire; the second does at 1080.
    assert t.wait_needed(7000) == pytest.approx(50.0)


def test_usage_older_than_a_minute_is_forgotten():
    clock = Clock()
    t = Throttle(8000, clock)
    t.record(8000)
    assert t.wait_needed(1) > 0
    clock.now += 60
    assert t.wait_needed(8000) == 0.0


def test_call_larger_than_the_limit_can_never_fit():
    assert Throttle(8000, Clock()).wait_needed(8001) is None


# --- decide (fast) ----------------------------------------------------------

def test_short_wait_stays_on_the_preferred_provider():
    assert decide(0.0, 30, is_last=False) == "go"
    assert decide(30.0, 30, is_last=False) == "go"


def test_long_wait_or_oversized_call_falls_back():
    assert decide(30.1, 30, is_last=False) == "skip"
    assert decide(None, 30, is_last=False) == "skip"


def test_last_provider_is_never_skipped():
    assert decide(55.0, 30, is_last=True) == "go"
    assert decide(None, 30, is_last=True) == "go"


def test_estimate_is_on_the_high_side():
    assert estimate_tokens([{"role": "user", "content": "x" * 3000}]) == 1000 + 800


# --- Real providers (llm) ---------------------------------------------------

MESSAGES = [{"role": "user", "content": "Reply with the single word: ok"}]


@pytest.fixture
def cfg():
    load_env()
    c = load_config()
    c.models.provider_order = ["groq", "openai"]
    return c


@pytest.mark.llm
def test_groq_is_used_first(cfg):
    result = LLM(cfg, "extract").chat(MESSAGES)
    assert result.provider == "groq"
    assert result.skipped == []
    assert result.response.choices[0].message.content


@pytest.mark.llm
def test_falls_back_to_openai_when_groq_pacing_wait_is_too_long(cfg):
    slept = []
    llm = LLM(cfg, "extract", sleep=slept.append)
    llm._throttles["groq"].record(cfg.models.groq.tokens_per_minute)   # budget already spent
    result = llm.chat(MESSAGES)
    assert result.provider == "openai"
    assert "pacing wait" in result.skipped[0]
    assert slept == []          # it did not wait on Groq first


@pytest.mark.llm
def test_waits_for_groq_when_the_wait_is_short(cfg):
    clock, slept = Clock(), []
    llm = LLM(cfg, "extract", sleep=slept.append, clock=clock)
    llm._throttles["groq"].record(cfg.models.groq.tokens_per_minute)
    clock.now += 45             # 15 seconds left in the window, under the 30s limit
    result = llm.chat(MESSAGES)
    assert result.provider == "groq"
    assert slept == [pytest.approx(15.0)]


@pytest.mark.llm
def test_falls_back_to_openai_when_groq_is_unreachable(cfg, monkeypatch):
    monkeypatch.setitem(PROVIDERS["groq"], "base_url", "http://127.0.0.1:9/v1")
    result = LLM(cfg, "extract").chat(MESSAGES)
    assert result.provider == "openai"
    assert result.skipped == ["groq: APIConnectionError"]
