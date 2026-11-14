"""Claim-level source check of chat answers (leadership/grounding.py).

The regression case is a real answer from manual testing (M6, 2026-11-13):
"Which former executives at Robinhood held finance-related leadership roles,
and when did they leave those roles?" The exact SQL and the exact answer are
copied from the chat log. Fast tests use no model; the `llm` test asks the
question for real, within a budget of 10 model calls.
"""

import re

import pytest

from leadership import sqlguard
from leadership.agent import Step
from leadership.config import ROOT
from leadership.grounding import problems

FIXTURE = ROOT / "fixtures" / "leadership.db"
COLLECTED = "2026-11-14"
QUESTION = ("Which former executives at Robinhood held finance-related leadership roles, and when did they leave "
            "those roles?")
SQL = ("SELECT person, title, canonical_role, ended_on, end_reported_on, ended_by_source_url, ended_by_source_date, "
       "source_url, source_type, source_date FROM leadership WHERE status = 'former' AND department = 'finance' "
       "ORDER BY person")
PROXY = "https://www.sec.gov/Archives/edgar/data/1783879/000178387926000053/hood-20260422.htm"
JUNE_8K = "https://www.sec.gov/Archives/edgar/data/1783879/000178387926000083/hood-20260625.htm"
FEB_8K = "https://www.sec.gov/Archives/edgar/data/1783879/000178387926000013/hood-20260206.htm"
SMEDLEY = "https://robinhood.com/newsroom/christina-smedley-joins-robinhood-as-chief-marketing-officer"

# The answer exactly as logged, including its non-breaking hyphens and narrow spaces.
BAD = ('- **Jason Warnick** – former Chief Financial Officer. He left the CFO role on **2026‑02‑06** '
       '(ended_on). This is reported in a Robinhood press release dated 2020‑08‑04: ' + SMEDLEY + '.\n\n'
       '- **Shiv Verma** – former Principal Accounting Officer. He left that position on **2026‑06‑25** '
       '(ended_on). The departure is documented in an SEC filing dated 2026‑06‑26: ' + JUNE_8K + '.\n\n'
       '- **Shiv Verma** – former Senior Vice President of Finance and Strategy, and Treasurer. No exact end '
       'date was given; the role was reported as ended on **2026‑04‑22** (end_reported_on). This information '
       'comes from an SEC filing dated 2022‑02‑10: ' + FEB_8K + '.')

GOOD = (f"- Jason Warnick – former Chief Financial Officer. He ceased to serve as CFO effective 2026-02-06 and has "
        f"left the company, according to an SEC filing, 2026-04-22: {PROXY}.\n\n"
        f"- Shiv Verma – former principal accounting officer. He ceased to serve in that role on 2026-06-25, "
        f"according to an SEC filing, 2026-06-26: {JUNE_8K}. He remains at Robinhood as Chief Financial "
        f"Officer, according to the same SEC filing, 2026-06-26: {JUNE_8K}.\n\n"
        f"- Shiv Verma – former Senior Vice President of Finance and Strategy, and Treasurer. The end date is not "
        f"stated; the role was first reported as ended in an SEC filing, 2026-04-22: {PROXY}.")


@pytest.fixture(scope="module")
def hood():
    return sqlguard.open_readonly(FIXTURE, "robinhood.com")


@pytest.fixture(scope="module")
def step(hood):
    r = sqlguard.run(hood, SQL)                       # the exact rows the agent received
    return Step(sql=SQL, columns=r.columns, rows=[list(x) for x in r.rows])


def test_the_logged_rows_are_the_three_former_finance_roles(step):
    people = [(row[0], row[1], row[3], row[4]) for row in step.rows]
    assert people == [("Jason Warnick", "Chief Financial Officer", "2026-02-06", None),
                      ("Shiv Verma", "principal accounting officer", "2026-06-25", None),
                      ("Shiv Verma", "Senior Vice President of Finance and Strategy, and Treasurer", None,
                       "2026-04-22")]


def test_the_logged_answer_fails_on_all_four_grounds(step, hood):
    found = problems(BAD, [step], hood, COLLECTED)
    joined = "\n".join(found)
    # 1. Warnick's end is cited to the 2020 press release, not to the proxy that shows it.
    assert f"Jason Warnick: the end of the role is cited to {SMEDLEY}" in joined and PROXY in joined
    # 2. Verma left the accounting role but is still at Robinhood (as CFO); the answer does not say so.
    assert "not that Shiv Verma is still at the company" in joined
    # 3. The end of the SVP role is cited to the February 8-K, not to the proxy that reported it.
    assert f"Shiv Verma: the end of the role is cited to {FEB_8K}" in joined
    # 4. 2022-02-10 is in no row (the filing is dated 2026-02-10).
    assert "the date 2022-02-10 is in none of the rows" in joined
    assert len(found) == 4


def test_the_corrected_answer_passes(step, hood):
    assert problems(GOOD, [step], hood, COLLECTED) == []


def test_a_report_date_given_as_the_end_date_fails(step, hood):
    text = (f"Shiv Verma – former Senior Vice President of Finance and Strategy, and Treasurer. He left that role on "
            f"2026-04-22, according to an SEC filing, 2026-04-22: {PROXY}. He remains at Robinhood as Chief "
            f"Financial Officer, according to an SEC filing, 2026-06-26: {JUNE_8K}.")
    found = problems(text, [step], hood, COLLECTED)
    assert found == ["2026-04-22 is given as the date a role ended, but it is end_reported_on, the date a source "
                     "reported the end; say the end date is not stated"]


def test_a_source_given_the_wrong_date_fails(step, hood):
    text = f"Jason Warnick ceased to serve as CFO effective 2026-02-06, according to an SEC filing, 2026-06-26: {PROXY}."
    assert problems(text, [step], hood, COLLECTED) == [
        f"2026-06-26 is given as the date of {PROXY}, but the data dates it 2026-04-22"]


def test_no_end_source_is_required_without_an_end_claim(step, hood):
    # Naming a former title and citing where the role was seen makes no claim about the end.
    text = f"Jason Warnick was Chief Financial Officer, according to a press release, 2020-08-04: {SMEDLEY}."
    assert problems(text, [step], hood, COLLECTED) == []


def test_valid_dates_from_other_supporting_rows_and_the_collection_date_pass(hood):
    r = sqlguard.run(hood, "SELECT person, title, source_url, source_date FROM current_leadership "
                           "WHERE person = 'Shiv Verma'")
    rows = Step(sql="x", columns=r.columns, rows=[list(x) for x in r.rows])
    text = ("Shiv Verma is currently Chief Financial Officer, according to an SEC filing, 2026-06-26: "
            f"{JUNE_8K}. Fortune profiled him on 2025-12-29. The data was collected on {COLLECTED}.")
    assert problems(text, [rows], hood, COLLECTED) == []


def test_answers_without_people_are_not_checked(hood):
    assert problems("0 vice presidents found in public sources. As of 2031-01-01.", [], hood) == []


def test_a_still_employed_claim_needs_a_current_source(step, hood):
    text = f"Shiv Verma remains at Robinhood as Chief Financial Officer, according to a press release: {SMEDLEY}."
    found = problems(text, [step], hood, COLLECTED)
    assert any(p.startswith("Shiv Verma: the answer says Shiv Verma is still at the company but cites no source "
                            f"of a current role; the current role in the data is Chief Financial Officer, source "
                            f"{JUNE_8K}") for p in found)


# ------------------------------------------------------------------ real model, at most 10 calls
@pytest.mark.llm
def test_real_answer_to_the_logged_question_is_grounded(tmp_path):
    from leadership.agent import Agent
    from leadership.config import load_config, load_env
    from leadership.grounding import grounded_ask
    from leadership.runlog import RunLog
    load_env()
    agent = Agent(load_config(), "robinhood.com", FIXTURE, log=RunLog(tmp_path / "chat.jsonl"))
    calls = []
    real_chat = agent.llm.chat

    def counted(*a, **k):
        calls.append(1)
        assert len(calls) <= 10, "more than 10 model calls"
        return real_chat(*a, **k)

    agent.llm.chat = counted
    answer = grounded_ask(agent, QUESTION)
    print(f"\nmodel calls: {len(calls)}\n{answer.text}")
    assert answer.ok and "Warning:" not in answer.text
    assert problems(answer.text, answer.steps, agent.conn, COLLECTED) == []
    assert "2022" not in answer.text and "Warnick" in answer.text and "Verma" in answer.text
    # Claim by claim: Warnick's end date is cited to the proxy, never to the 2020 press release. The press release
    # may still be cited for "held the CFO role", which it supports.
    end = next(s for s in re.split(r"(?<=\.)\s+(?=[A-Z])", answer.text.replace("‑", "-")) if "2026-02-06" in s)
    assert PROXY in end and SMEDLEY not in end
    assert re.search(r"Verma[^\n]*\b(still|remains)\b", answer.text)


def test_a_source_of_another_person_fails(step, hood):
    tenev_only = "https://investors.robinhood.com/management/vlad-tenev"
    text = f"Jason Warnick was Chief Financial Officer, according to {tenev_only}."
    assert problems(text, [step], hood, COLLECTED) == [
        f"Jason Warnick: the answer cites {tenev_only}, which is not a source of any of Jason Warnick's rows"]


MAY_8K = "https://www.sec.gov/Archives/edgar/data/1783879/000178387926000065/hood-20260507.htm"


def test_pinner_end_cited_to_the_proxy_fails_and_the_corrected_answer_passes(hood):
    """rh08 in the chat evaluation. The April proxy lists Pinner as CTO; his separation on 2026-05-07 is in the
    May 8-K (filed 2026-05-08)."""
    r = sqlguard.run(hood, "SELECT person, title, status, ended_on, ended_by_source_url, ended_by_source_date, "
                           "source_url, source_date FROM leadership WHERE person = 'Jeffrey Pinner'")
    rows = Step(sql="x", columns=r.columns, rows=[list(x) for x in r.rows])
    cached = ("No. Jeffrey Pinner is no longer CTO; the role is marked former and ended on 2026-05-07, according to "
              f"a sec_filing, 2026-04-22: {PROXY}.")
    assert problems(cached, [rows], hood, COLLECTED)[0].startswith(
        f"Jeffrey Pinner: the end of the role is cited to {PROXY}")
    # The corrected answer from the real correction turn: the end cited to the 8-K, the role itself to the proxy.
    # "<url>. ... sec_filing, 2026-04-22: <proxy>" must not pair 2026-04-22 with the 8-K.
    corrected = ("No. Jeffrey Pinner is no longer CTO. The role is marked former and ended on 2026-05-07, according to "
                 f"the sec_filing, 2026-05-08: {MAY_8K}. The role row itself was sourced from the sec_filing, "
                 f"2026-04-22: {PROXY}.")
    assert problems(corrected, [rows], hood, COLLECTED) == []
    assert problems(f"- Jeffrey Pinner: {MAY_8K} (sec_filing, 2026-04-22)", [rows], hood, COLLECTED) == [
        f"2026-04-22 is given as the date of {MAY_8K}, but the data dates it 2026-05-08"]


def test_the_evaluation_grader_runs_the_source_check(hood):
    from leadership.agent import Answer
    from leadership.chateval import grade
    sql = ("SELECT person, title, status, ended_on, ended_by_source_url, source_url, source_date FROM leadership "
           "WHERE person = 'Jeffrey Pinner'")
    r = sqlguard.run(hood, sql)
    step = Step(sql=sql, columns=r.columns, rows=[list(x) for x in r.rows])
    case = {"id": "rh08", "question": "Is Jeffrey Pinner still the CTO?", "check": "names",
            "sql": "SELECT DISTINCT person FROM leadership WHERE person = 'Jeffrey Pinner'"}
    bad = Answer(case["question"], f"No. Jeffrey Pinner is no longer CTO; it ended on 2026-05-07, according to a "
                                   f"sec_filing, 2026-04-22: {PROXY}.", [step], ok=True)
    g = grade(case, bad, hood, COLLECTED)
    assert not g.passed and g.grounding and g.problems[-1].startswith("source check: Jeffrey Pinner")
    good = Answer(case["question"], f"No. Jeffrey Pinner is no longer CTO; it ended on 2026-05-07, according to a "
                                    f"sec_filing, 2026-05-08: {MAY_8K}.", [step], ok=True)
    assert grade(case, good, hood, COLLECTED).passed
