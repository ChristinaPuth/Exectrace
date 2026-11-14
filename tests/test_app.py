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


def test_without_a_key_the_page_says_what_to_do_and_asks_nothing(no_keys):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert "No LLM key found" in at.error[0].value
    assert not at.chat_input                          # no question can be asked without a model


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
