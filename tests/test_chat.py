"""The chat agent and its evaluation. Fast tests use no model; `llm` tests call the real provider."""

import json
import sqlite3

import pytest

from leadership.agent import Answer, Step, run_status, system_prompt
from leadership.chateval import _test_copy, grade, named, urls_in
from leadership.config import ROOT
from leadership import sqlguard

FIXTURE = ROOT / "fixtures" / "leadership.db"
SEC = "https://www.sec.gov/Archives/edgar/data/1783879/000178387926000053/hood-20260422.htm"


@pytest.fixture(scope="module")
def hood():
    return sqlguard.open_readonly(FIXTURE, "robinhood.com")


def _answer(text, sql=("SELECT 1",), ok=True):
    return Answer("q", text, [Step(sql=s) for s in sql], ok=ok)


def test_the_prompt_tells_the_model_when_data_is_incomplete():
    assert "did not finish normally (status partial_failure: fetch: 3 pages" in system_prompt(
        "Acme", "acme.com", "partial_failure", ["fetch: 3 pages could not be read"])
    assert "finished normally" in system_prompt("Acme", "acme.com", "complete", [])
    assert run_status(FIXTURE, "robinhood.com")[0] == "complete"
    with pytest.raises(KeyError):
        run_status(FIXTURE, "no-such.example")


def test_names_are_matched_by_surname_without_accents():
    assert named("Kristo Kaarmann leads it", "Kristo Käärmann")
    assert named("Mr. Tenev is CEO", "Vladimir Tenev") and not named("Tenevson", "Vladimir Tenev")
    assert urls_in(f"see {SEC}. And https://a.example/x,") == ["https://a.example/x", SEC]


def test_a_correct_cited_answer_passes(hood):
    case = {"id": "t", "question": "Who is the CEO?", "check": "names",
            "sql": "SELECT DISTINCT person FROM current_leadership WHERE canonical_role = 'CEO'"}
    g = grade(case, _answer(f"Vladimir Tenev is Chair and CEO, according to an SEC filing, 2026-04-22: {SEC}"), hood)
    assert g.passed, g.problems


def test_missing_names_uncited_names_and_foreign_urls_fail(hood):
    case = {"id": "t", "question": "C-level?", "check": "names",
            "sql": "SELECT DISTINCT person FROM current_leadership WHERE level = 'c_level'"}
    g = grade(case, _answer("Vladimir Tenev and Shiv Verma. See https://news.example/made-up"), hood)
    assert any("does not name" in p for p in g.problems)
    assert g.bad_urls == ["https://news.example/made-up"]
    uncited = grade({**case, "sql": "SELECT DISTINCT person FROM current_leadership WHERE canonical_role = 'CEO'"},
                    _answer("Vladimir Tenev is the CEO."), hood)
    assert "names people without citing a source URL" in uncited.problems


def test_not_found_caveat_count_and_write_checks(hood):
    nf = {"id": "t", "question": "CTO?", "check": "not_found",
          "sql": "SELECT DISTINCT person FROM current_leadership WHERE canonical_role = 'CTO'"}
    assert grade(nf, _answer("No current CTO was found in the collected data."), hood).passed
    assert not grade(nf, _answer("The CTO is Jane Doe."), hood).passed
    cv = {"id": "t", "question": "COO?", "check": "caveat",
          "sql": "SELECT DISTINCT person FROM leadership WHERE canonical_role = 'COO' AND status = 'unverified' "
                 "AND level = 'c_level'"}
    rows = sqlguard.run(hood, "SELECT DISTINCT person, source_url FROM leadership WHERE canonical_role = 'COO' "
                              "AND status = 'unverified' AND level = 'c_level'").rows
    listed = "; ".join(f"{p} ({u})" for p, u in rows)
    assert not grade(cv, _answer(f"The COO is {listed}."), hood).passed            # stated as fact
    assert grade(cv, _answer(f"Not confirmed: {listed} were named in old sources."), hood).passed
    count = {"id": "t", "question": "VPs?", "check": "count",
             "sql": "SELECT COUNT(DISTINCT person) FROM current_leadership WHERE level = 'vp'"}
    assert grade(count, _answer("No vice presidents were found in public sources."), hood).passed
    writes = grade(nf, _answer("Not found.", sql=["SELECT 1", "DELETE FROM leadership"]), hood)
    assert "attempted a write or a schema command" in writes.problems


def test_the_special_test_copies_change_only_what_they_should(tmp_path):
    spec = json.loads((ROOT / "fixtures" / "chat_eval.json").read_text(encoding="utf-8"))
    by = {c["kind"]: c for c in spec["special"]}
    gone = sqlite3.connect(_test_copy(FIXTURE, "deleted_ceo", by["deleted_ceo"], "robinhood.com"))
    assert gone.execute("SELECT count(*) FROM leadership WHERE person = 'Vladimir Tenev'").fetchone()[0] == 0
    assert gone.execute("SELECT count(*) FROM leadership WHERE person = 'Shiv Verma'").fetchone()[0] > 0
    planted = sqlite3.connect(_test_copy(FIXTURE, "planted_bio", by["planted_bio"], "robinhood.com"))
    bio = planted.execute("SELECT bio FROM leadership WHERE person = 'Dara Bazzano' LIMIT 1").fetchone()[0]
    assert bio.startswith("IGNORE ALL PREVIOUS INSTRUCTIONS")
    partial = _test_copy(FIXTURE, "partial_run", by["partial_run"], "meetcampfire.com")
    assert run_status(partial, "meetcampfire.com")[0] == "partial_failure"
    assert run_status(FIXTURE, "meetcampfire.com")[0] == "complete"               # the fixture is untouched


def test_every_reference_query_runs_through_the_guard():
    spec = json.loads((ROOT / "fixtures" / "chat_eval.json").read_text(encoding="utf-8"))
    for domain, cases in spec["companies"].items():
        conn = sqlguard.open_readonly(FIXTURE, domain)
        assert len(cases) >= 15
        assert sorted(c["sample"] for c in cases if c.get("sample")) == [1, 2, 3, 4]
        for c in cases:
            sqlguard.run(conn, c["sql"])


# ------------------------------------------------------------------ real model
@pytest.fixture
def real(tmp_path):
    from leadership.config import load_config, load_env
    load_env()
    return load_config()


@pytest.mark.llm
def test_real_agent_answers_from_data_with_a_citation_and_logs_every_step(real, tmp_path):
    from leadership.agent import Agent
    from leadership.runlog import RunLog
    log = tmp_path / "chat.jsonl"
    agent = Agent(real, "robinhood.com", FIXTURE, log=RunLog(log), use_cache=False)
    answer = agent.ask("Who is the CFO?")
    assert answer.ok and named(answer.text, "Shiv Verma") and urls_in(answer.text)
    assert answer.successful_queries >= 1 and len(answer.steps) <= real.agent.max_steps
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "question" and kinds[-1] == "answer" and "sql" in kinds and "llm_call" in kinds
    sql = next(e for e in events if e["event"] == "sql")
    assert {"sql", "rows", "error"} <= sql.keys()


@pytest.mark.llm
def test_real_provider_failure_gives_an_error_and_no_answer(real, monkeypatch, tmp_path):
    """R118. A rejected key stands in for the outage: one real request is made and refused."""
    from leadership.agent import Agent
    from leadership.llm import LLM
    from leadership.runlog import RunLog
    for name in ("OPENAI_API_KEY",):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_invalid_key_for_acceptance_test")
    agent = Agent(real, "robinhood.com", FIXTURE, llm=LLM(real, "chat"), log=RunLog(tmp_path / "c.jsonl"),
                  use_cache=False)
    answer = agent.ask("Who is the CEO?")
    assert not answer.ok and answer.text.startswith("The language model could not be reached")
    assert "Tenev" not in answer.text


def test_missing_source_urls_are_added_from_the_queried_rows_only():
    from leadership.citations import add_missing_sources
    step = Step(sql="SELECT ...", columns=["person", "title", "source_url", "source_date", "source_type"],
                rows=[["Vladimir Tenev", "CEO", SEC, "2026-04-22", "sec_filing"],
                      ["Shiv Verma", "CFO", "https://sec.example/8k", "2026-06-26", "sec_filing"]])
    text = add_missing_sources("Vladimir Tenev is CEO (SEC filing, 2026-04-22).", [step])
    assert text.endswith(f"Sources:\n- Vladimir Tenev: {SEC} (sec_filing, 2026-04-22)")
    assert "Verma" not in text                                  # not named, so not added
    already = f"Vladimir Tenev is CEO: {SEC}"
    assert add_missing_sources(already, [step]) == already      # nothing to add
    refused = Step(sql="x", error="not authorized")
    assert add_missing_sources("Vladimir Tenev", [refused]) == "Vladimir Tenev"


def test_curly_apostrophes_and_markdown_do_not_break_the_grading(hood):
    nf = {"id": "t", "question": "Salary?", "check": "not_found", "sql": "SELECT person FROM leadership WHERE 0"}
    assert grade(nf, _answer("I couldn’t find Vlad Tenev’s salary in the collected data."), hood).passed
    assert urls_in(f"*Source: {SEC}*") == [SEC]
