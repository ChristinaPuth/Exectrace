"""The Streamlit page, run headless with Streamlit's own test runner. No browser.

Fast tests need no key and make no model call. The `llm` test asks one
question; with the evaluation's cache in place it is answered from the
cache, otherwise one real question is asked.
"""

import pytest
from streamlit.testing.v1 import AppTest

from leadership.config import ROOT

APP = str(ROOT / "app.py")


@pytest.fixture
def no_keys(monkeypatch):
    for name in ("OPENAI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(name, "")                  # load_env never overrides a variable that is set


def test_the_page_lists_the_companies_in_the_fixtures(no_keys):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Company leadership"
    box = at.sidebar.selectbox[0]
    assert box.label == "Company" and set(box.options) >= {"Campfire (meetcampfire.com)",
                                                           "Robinhood (robinhood.com)"}


def test_without_a_server_key_the_page_asks_for_the_visitors_key_and_nothing_else(no_keys):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "Enter a Groq or OpenAI key in the sidebar" in at.info[0].value
    assert at.sidebar.text_input[0].label == "Groq API key"
    assert not at.chat_input                          # no question can be asked without a checked key


def _fake_check(calls):
    def check(cfg, provider, key):
        calls.append((provider, key))
        return (key == "gsk_good_test_key"), ("The key works." if key == "gsk_good_test_key" else "Rejected.")
    return check


def test_a_checked_visitor_key_opens_the_chat_and_never_reaches_the_environment(no_keys, monkeypatch):
    import os
    import leadership.llm
    calls = []
    monkeypatch.setattr(leadership.llm, "check_key", _fake_check(calls))   # no network in fast tests
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.text_input[0].set_value("gsk_bad_test_key").run()
    at.sidebar.button[0].click().run()
    assert at.sidebar.error[0].value == "Rejected." and not at.chat_input
    at.sidebar.text_input[0].set_value("gsk_good_test_key").run()
    at.sidebar.button[0].click().run()
    assert not at.exception and at.sidebar.success[0].value == "The key works."
    assert at.chat_input[0].placeholder == "Ask about Campfire's leadership"
    assert calls == [("groq", "gsk_bad_test_key"), ("groq", "gsk_good_test_key")]
    # The key is for this session's agent only: not in the environment, not stored as a fingerprint.
    assert os.environ["GROQ_API_KEY"] == "" and "gsk_good_test_key" not in at.session_state["checked"]
    agent = next(iter(at.session_state["agents"].values()))
    assert agent.llm.chain[0][0] == "groq" and agent.llm._keys == {"groq": "gsk_good_test_key"}
    # Changing the key closes the chat until the new one is checked.
    at.sidebar.text_input[0].set_value("gsk_other_test_key").run()
    assert not at.chat_input and "The key changed" in at.sidebar.info[0].value


def test_the_key_check_needs_a_key_and_a_known_provider():
    from leadership.config import load_config
    from leadership.llm import check_key
    assert check_key(load_config(), "groq", " ") == (False, "Enter a key first.")
    assert check_key(load_config(), "acme", "x")[0] is False


@pytest.mark.llm
def test_real_key_check_and_a_question_with_the_visitors_key(no_keys):
    """A rejected key (one request, no tokens), then the real Groq key from .env typed in as a visitor would."""
    from dotenv import dotenv_values
    from leadership.config import load_config
    from leadership.llm import check_key
    ok, message = check_key(load_config(), "groq", "gsk_invalid_key_for_acceptance_test")
    assert not ok and message == "Groq rejected this key."
    key = dotenv_values(ROOT / ".env").get("GROQ_API_KEY")
    if not key:
        pytest.skip("no GROQ_API_KEY in .env")
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.selectbox[0].select("robinhood.com").run()
    at.sidebar.text_input[0].set_value(key).run()
    at.sidebar.button[0].click().run()
    assert at.sidebar.success[0].value.startswith("The key works")
    at.chat_input[0].set_value("Who is the CEO?").run()
    assert not at.exception and "Tenev" in " ".join(m.value for m in at.markdown)


@pytest.mark.llm
def test_a_question_shows_the_answer_with_its_sql_and_rows():
    from leadership.config import load_env
    load_env()
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.selectbox[0].select("robinhood.com").run()
    at.chat_input[0].set_value("Who is the CEO?").run()
    assert not at.exception
    answer = " ".join(m.value for m in at.markdown)
    assert "Tenev" in answer and "https://" in answer
    assert at.expander[0].label == "SQL and source rows"
    assert "current_leadership" in at.code[0].value
    assert any(m.value.startswith("| person |") for m in at.markdown)       # the source rows as a table
    # A second company keeps its own history.
    at.sidebar.selectbox[0].select("meetcampfire.com").run()
    assert not any("Tenev" in m.value for m in at.markdown)


@pytest.mark.llm
def test_a_count_on_the_page_says_what_it_can_miss():
    """M5 review fix 10, on the page: "0 found" is not shown as "there are none"."""
    from leadership.config import load_env
    load_env()
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.selectbox[0].select("robinhood.com").run()
    at.chat_input[0].set_value("How many vice presidents were found?").run()
    assert not at.exception
    answer = " ".join(m.value for m in at.markdown)
    assert "could not be read" in answer and "None found does not mean there are none" in answer
