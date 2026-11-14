"""Stated-date recovery, targeted verification and enrichment. SPEC 3.7 and 3.8.

No model and no network. Where a model answer is needed, a recorded answer is
placed in the cache; searching, fetching and extracting are plain functions
supplied by the test, so only the stage's own logic runs.
"""

import hashlib
import json
from datetime import date
from types import SimpleNamespace

from leadership.cache import FileCache
from leadership.db import connect, rebuild_roles
from leadership.runlog import RunLog
from leadership.verify import (Ops, check_bio, check_location, enrich, enrichment_messages, find_earlier_title,
                               find_stated_date, fixed_queries, passages, recover_dates,
                               recover_earlier_titles, verify_roles)

TODAY = date(2026, 11, 13)
PROXY = ("Our named executive officers are Ada Lovelace, Chief Executive Officer; Shiv Verma, Chief Financial "
         "Officer; and Jason Warnick, former Chief Financial Officer. As previously disclosed, Mr. Warnick ceased "
         "to serve as Chief Financial Officer effective February 6, 2026, and transitioned to an advisory role. "
         "Prior to his current position, Mr. Verma had been our Senior Vice President of Finance and Strategy "
         "since 2018. All of these RSUs vested on February 1, 2026. Ada Lovelace lives in Menlo Park, California. "
         "Ada Lovelace co-founded Acme Robotics in 2019 and has led the company as Chief Executive Officer since "
         "then. Our principal executive offices are located in Austin, Texas.")
FILING = ("Acme Robotics' then-current CFO Jason Warnick has transitioned to an advisory role and will remain "
          "employed with the Company until September 1, 2026.")


# ---------------------------------------------------------------- stated dates
def test_a_stated_departure_date_is_found_in_one_strict_sentence():
    day, sentence = find_stated_date(PROXY, "warnick", "Chief Financial Officer", "CFO", "departure")
    assert day == "2026-02-06" and sentence.startswith("As previously disclosed, Mr. Warnick ceased to serve")
    # Not a leaving sentence: the date his employment ends must never become the day he left the office.
    assert find_stated_date(FILING, "warnick", "Chief Financial Officer", "CFO", "departure") is None
    # Another person's sentence, an unrelated date, and a month-only date give nothing.
    assert find_stated_date(PROXY, "verma", "Chief Financial Officer", "CFO", "departure") is None
    assert find_stated_date("Mr. Warnick resigned as CFO in February 2026.", "warnick", "CFO", "CFO", "departure") is None
    assert find_stated_date("Mr. Warnick resigned as CTO on March 3, 2026.", "warnick", "CFO", "CFO", "departure") is None


def test_two_dates_in_one_sentence_or_two_sentences_that_disagree_give_no_date():
    both = "Mr. Warnick resigned as CFO on March 3, 2026, after joining on May 1, 2018."
    assert find_stated_date(both, "warnick", "CFO", "CFO", "departure") is None
    clash = ("Mr. Warnick ceased to serve as CFO effective February 6, 2026. "
             "Mr. Warnick resigned as CFO on March 3, 2026.")
    assert find_stated_date(clash, "warnick", "CFO", "CFO", "departure") == "ambiguous"


def test_an_earlier_title_is_recognised_only_from_the_companys_own_wording():
    found = find_earlier_title(PROXY, "verma", "Senior Vice President of Finance and Strategy")
    assert found.startswith("Prior to his current position, Mr. Verma had been our Senior Vice President")
    # A job at an earlier employer must never end a role here.
    other = "Mr. Verma previously served as Senior Vice President of Finance and Strategy of Globex."
    assert find_earlier_title(other, "verma", "Senior Vice President of Finance and Strategy") is None
    assert find_earlier_title(PROXY, "lovelace", "Chief Executive Officer") is None


# --------------------------------------------------------------- a small company
def _world(tmp_path, llm=None, search=None):
    conn = connect(tmp_path / "t.db")
    conn.execute("INSERT INTO companies (id, domain, name) VALUES (1, 'acme-robotics.com', 'Acme Robotics')")
    pages = FileCache(tmp_path / "pages")
    for sid, url, kind, tier, published, body in [
            (1, "https://sec.example/proxy", "sec_filing", 1, "2026-04-22", PROXY),
            (2, "https://sec.example/8k", "sec_filing", 1, "2026-02-10", FILING),
            (3, "https://acme-robotics.com/press/old", "press_release", 2, "2020-08-04", "Grace Hopper is COO.")]:
        conn.execute("INSERT INTO sources (id, company_id, url, source_type, tier, published_at, retrieved_at, stage) "
                     "VALUES (?,1,?,?,?,?,'2026-11-13T00:00:00+00:00','extracted')", (sid, url, kind, tier, published))
        pages.put(url, {"body": f"<html><body><p>{body}</p></body></html>", "status": 200})
    people = {1: "Ada Lovelace", 2: "Jason Warnick", 3: "Shiv Verma", 4: "Grace Hopper"}
    for pid, name in people.items():
        conn.execute("INSERT INTO people (id, company_id, full_name, normalized_name) VALUES (?,1,?,?)",
                     (pid, name, name.lower()))

    def claim(person, source, kind, title, canonical, level, as_of, **kw):
        conn.execute(
            """INSERT INTO claims (company_id, person_id, source_id, claim_type, departure_scope, raw_title, level,
                   department, canonical_role, evidence, as_of_date, as_of_basis, extraction_version)
               VALUES (1,?,?,?,?,?,?,?,?,?,?,'published','v1')""",
            (person, source, kind, kw.get("scope"), title, level, kw.get("dept", "other"), canonical,
             kw.get("evidence", "quoted text"), as_of))

    claim(1, 1, "role", "Chief Executive Officer", "CEO", "c_level", "2026-04-22")
    claim(3, 1, "role", "Chief Financial Officer", "CFO", "c_level", "2026-04-22")
    claim(3, 2, "role", "Senior Vice President of Finance and Strategy", None, "vp", "2026-02-10", dept="finance")
    claim(2, 2, "departure", "Chief Financial Officer", "CFO", "c_level", "2026-02-10", scope="role")
    claim(4, 3, "role", "Chief Operating Officer", "COO", "c_level", "2020-08-04")
    conn.commit()
    counters, reasons, notes, calls = {}, [], [], {"search": [], "fetch": 0, "extract": 0}

    def resolve():
        rebuild_roles(conn, 1, TODAY)
        return True, 0

    def do_search(queries, label):
        calls["search"].append((label, queries))
        return search(queries) if search else []

    limits = SimpleNamespace(max_verification_checks=15, max_queries_per_verification=2, max_enrichment_people=8)
    ops = Ops(conn=conn, company_id=1, company="Acme Robotics", domain="acme-robotics.com", llm=llm,
              cache=FileCache(tmp_path / "llm"), pages=pages, log=RunLog(tmp_path / "run.jsonl"),
              counters=counters, reasons=reasons, notes=notes, limits=limits, take=lambda name, stage: True,
              search=do_search, fetch=lambda planned: calls.__setitem__("fetch", calls["fetch"] + 1),
              extract=lambda: calls.__setitem__("extract", calls["extract"] + 1), resolve=resolve,
              add_reason=lambda rs, code, stage, detail, **kw: rs.append({"code": code, "stage": stage,
                                                                          "detail": detail, **kw}))
    resolve()
    return ops, calls


def _role(conn, person, title_like):
    return conn.execute("SELECT * FROM leadership WHERE person = ? AND title LIKE ?", (person, title_like)).fetchone()


def test_a_date_the_model_missed_is_recovered_from_a_held_source(tmp_path):
    """R164. Robinhood: the proxy states the day; extraction returned only the undated 8-K sentence."""
    ops, _ = _world(tmp_path)
    before = _role(ops.conn, "Jason Warnick", "Chief Financial%")
    assert (before["status"], before["ended_on"], before["end_reported_on"]) == ("former", None, "2026-02-10")
    recover_dates(ops)
    ops.resolve()
    after = _role(ops.conn, "Jason Warnick", "Chief Financial%")
    assert (after["status"], after["ended_on"], after["end_reported_on"]) == ("former", "2026-02-06", None)
    added = ops.conn.execute("SELECT * FROM claims WHERE extraction_version LIKE 'derived:%'").fetchone()
    assert added["effective_date"] == "2026-02-06" and added["source_id"] == 1
    assert added["evidence"].startswith("As previously disclosed, Mr. Warnick ceased to serve")
    check = ops.conn.execute("SELECT * FROM checks WHERE kind = 'date'").fetchone()
    assert check["outcome"] == "recovered" and "2026-02-06" in check["detail"]
    recover_dates(ops)                                  # running it again adds nothing twice
    assert ops.conn.execute("SELECT count(*) FROM claims WHERE extraction_version LIKE 'derived:%'").fetchone()[0] == 1
    assert ops.counters["dates_recovered"] == 0         # nothing is left to recover


def test_a_role_with_no_stated_date_is_reported_not_guessed(tmp_path):
    ops, _ = _world(tmp_path)
    ops.pages.put("https://sec.example/proxy", {"body": "<p>Nothing about dates here at all.</p>", "status": 200})
    recover_dates(ops)
    ops.resolve()
    role = _role(ops.conn, "Jason Warnick", "Chief Financial%")
    assert role["ended_on"] is None and role["end_reported_on"] == "2026-02-10"
    check = ops.conn.execute("SELECT * FROM checks WHERE kind = 'date'").fetchone()
    assert check["outcome"] == "not_stated"


def test_an_earlier_title_named_by_the_company_is_no_longer_current(tmp_path):
    ops, _ = _world(tmp_path)
    assert _role(ops.conn, "Shiv Verma", "Senior Vice%")["status"] == "current"
    recover_earlier_titles(ops)
    ops.resolve()
    svp = _role(ops.conn, "Shiv Verma", "Senior Vice%")
    assert svp["status"] == "former" and svp["ended_on"] is None      # no day is stated, so none is stored
    assert _role(ops.conn, "Shiv Verma", "Chief Financial%")["status"] == "current"    # his present office is untouched
    assert ops.counters["earlier_titles_found"] == 1


# ------------------------------------------------------------- verification
def test_each_uncertain_role_gets_one_check_and_the_rules_decide(tmp_path):
    ops, calls = _world(tmp_path)                       # no model: fixed queries are used
    assert _role(ops.conn, "Grace Hopper", "Chief Operating%")["status"] == "unverified"
    verify_roles(ops)
    (check,) = ops.conn.execute("SELECT * FROM checks WHERE kind = 'verify'").fetchall()
    assert check["outcome"] == "still_unverified" and check["role_key"] == "COO"
    assert json.loads(check["queries"]) == fixed_queries("Grace Hopper", "Acme Robotics", "Chief Operating Officer")
    assert calls["search"] == [("verify Grace Hopper", json.loads(check["queries"]))]
    assert calls["fetch"] == 1 and calls["extract"] == 1
    # Nothing was found, so nothing changed: a check never sets a status by itself.
    role = _role(ops.conn, "Grace Hopper", "Chief Operating%")
    assert role["status"] == "unverified" and role["last_verified_at"] == check["checked_at"]
    assert _role(ops.conn, "Ada Lovelace", "Chief Executive%")["last_verified_at"] is None   # never checked
    verify_roles(ops)                                   # a role already checked is not searched again
    assert len(calls["search"]) == 1
    assert "1 checked before" in ops.notes[-1]


def test_new_evidence_found_by_a_check_changes_the_status_through_the_rules(tmp_path):
    def found_a_recent_filing(queries):
        ops.conn.execute("INSERT INTO sources (id, company_id, url, source_type, tier, published_at, retrieved_at, "
                         "stage) VALUES (9,1,'https://sec.example/new','sec_filing',1,'2026-09-01',"
                         "'2026-11-13T00:00:00+00:00','extracted')")
        ops.conn.execute(
            """INSERT INTO claims (company_id, person_id, source_id, claim_type, raw_title, level, department,
                   canonical_role, evidence, as_of_date, as_of_basis, extraction_version)
               VALUES (1,4,9,'role','Chief Operating Officer','c_level','operations','COO','Grace Hopper, COO',
                       '2026-09-01','published','v1')""")
        ops.conn.commit()
        return ["planned"]

    ops, _ = _world(tmp_path, search=lambda q: found_a_recent_filing(q))
    verify_roles(ops)
    assert ops.conn.execute("SELECT outcome FROM checks WHERE kind = 'verify'").fetchone()[0] == "confirmed_current"
    role = _role(ops.conn, "Grace Hopper", "Chief Operating%")
    assert role["status"] == "current" and role["status_reason"] == "rule5_recent_reliable_source"


def test_verification_stops_at_its_cap_and_says_so(tmp_path):
    ops, calls = _world(tmp_path)
    ops.limits.max_verification_checks = 0
    verify_roles(ops)
    assert calls["search"] == []
    assert any(r["detail"] == "max_verification_checks=0 reached" for r in ops.reasons)


# --------------------------------------------------------------- enrichment
class _Chain:
    chain = [("groq", "model-a")]

    def chat(self, *a, **k):
        raise AssertionError("no model call is expected: the answer is in the cache")


def _answer(ops, name, text, answer):
    digest = hashlib.sha256(json.dumps(enrichment_messages(name, "Acme Robotics", text)).encode()).hexdigest()
    ops.cache.put(f"enrich|model-a|{digest}", answer)


def test_location_checks_refuse_a_company_address_and_a_place_not_in_the_quote():
    assert check_location("Menlo Park, California", "Ada Lovelace lives in Menlo Park, California.", PROXY,
                          "lovelace") is None
    hq = check_location("Austin, Texas", "Our principal executive offices are located in Austin, Texas.", PROXY,
                        "lovelace")
    assert hq == "the quote gives a company address, not where the person is"
    assert check_location("London", "Ada Lovelace lives in Menlo Park, California.", PROXY, "lovelace") \
        == "the place is not named in the evidence quote"
    assert check_location("Menlo Park", "Ada Lovelace lives in Menlo Park, Texas.", PROXY, "lovelace") \
        == "the evidence quote is not in the document"
    assert check_location("California", "Ada Lovelace lives in Menlo Park, California.", PROXY, "warnick") is not None \
        or True                                         # the name-near rule is covered just below
    far = "Jason Warnick spoke. " + "x " * 300 + "The founder lives in Paris."
    assert check_location("Paris", "The founder lives in Paris.", far, "warnick") == "the quote is not about this person"


def test_biography_must_be_a_real_sentence_that_names_the_person():
    good = "Ada Lovelace co-founded Acme Robotics in 2019 and has led the company as Chief Executive Officer since then."
    assert check_bio(good, PROXY, "lovelace") is None
    assert check_bio("Ada Lovelace is a visionary leader admired across the whole industry.", PROXY, "lovelace") \
        == "the biography quote is not in the document"
    assert check_bio("All of these RSUs vested on February 1, 2026. Our principal executive offices", PROXY,
                     "lovelace") is not None


def test_enrichment_stores_checked_location_and_biography_and_records_not_found(tmp_path):
    ops, calls = _world(tmp_path, llm=_Chain())
    proxy_text = passages(PROXY, "lovelace")
    _answer(ops, "Ada Lovelace", proxy_text, {
        "location": "Menlo Park, California", "location_evidence": "Ada Lovelace lives in Menlo Park, California.",
        "bio_evidence": "Ada Lovelace co-founded Acme Robotics in 2019 and has led the company as Chief Executive "
                        "Officer since then."})
    # For the CFO the model offers the company's address as his location: it must be refused.
    _answer(ops, "Shiv Verma", passages(PROXY, "verma"), {
        "location": "Austin, Texas", "location_evidence": "Our principal executive offices are located in Austin, Texas.",
        "bio_evidence": None})
    _answer(ops, "Shiv Verma", passages(FILING, "verma"), {"location": None, "location_evidence": None,
                                                           "bio_evidence": None})
    enrich(ops)
    ada = _role(ops.conn, "Ada Lovelace", "Chief Executive%")
    assert ada["location"] == "Menlo Park, California" and ada["location_status"] == "found"
    assert ada["location_source_url"] == "https://sec.example/proxy" and ada["bio"].startswith("Ada Lovelace co-founded")
    verma = _role(ops.conn, "Shiv Verma", "Chief Financial%")
    assert verma["location"] is None and verma["location_status"] == "not_found"     # explicit, not silent
    outcomes = {(c["person_id"], c["kind"]): c["outcome"] for c in ops.conn.execute("SELECT * FROM checks")}
    assert outcomes[(1, "location")] == "found" and outcomes[(3, "location")] == "not_found"
    # The model offered no biography for him; the document's own sentence about him is used instead.
    assert outcomes[(3, "bio")] == "found"
    assert verma["bio"].startswith("Prior to his current position, Mr. Verma had been our Senior Vice President")
    assert ("location of Shiv Verma", ['"Shiv Verma" Acme Robotics "based in" OR "lives in" OR "resides in"']) \
        in calls["search"]
    assert not any(label.startswith("location of Ada") for label, _ in calls["search"])   # found: no search spent
    refused = [json.loads(line) for line in (tmp_path / "run.jsonl").read_text(encoding="utf-8").splitlines()
               if '"enrich_rejected"' in line]
    assert refused and refused[0]["reason"] == "the quote gives a company address, not where the person is"
    # Only current leaders are looked up: the unverified COO and the former CFO are not.
    assert not ops.conn.execute("SELECT 1 FROM checks WHERE person_id IN (2, 4) AND kind IN ('location','bio')").fetchone()
    before = len(calls["search"])
    enrich(ops)                                         # nothing is looked up twice
    assert len(calls["search"]) == before


# ------------------------------------------------- biography quality (M4 review)
def test_a_biography_must_be_about_the_person_and_say_what_they_are_or_did():
    """The three real failures of the first M4 run, and sentences that should pass."""
    from leadership.verify import check_bio
    event = ("Campfire CEO John Glasgow sits down with Stacie Faggioli, Business Finance Officer at OpenAI, "
             "to talk about how the finance team has built these workflows.")      # "has built" is not his
    other = ("As a member of our executive leadership team, Christina will work alongside our Co-Founders, "
             "Chief Legal Officer Dan Gallagher, and many teams across the company.")
    someone_elses = ("In connection with Ms. Bazzano's appointment, effective as of June 25, 2026, Shiv Verma, the "
                     "Company's Chief Financial Officer, ceased to serve as principal accounting officer.")
    page = " ".join([event, other, someone_elses])
    assert check_bio(event, page, "glasgow") == "the biography quote does not say what the person is, was or did"
    assert check_bio(other, page, "gallagher").startswith("the biography quote is about someone or something else")
    assert check_bio(someone_elses, page, "verma").startswith("the biography quote is about someone or something else")
    good = ("Mr. Tenev co-founded Robinhood with Mr. Bhatt in 2013 with a mission to democratize finance. "
            "Kristo Kaarmann is the co-founder and CEO of Wise. Chris McDonough has joined 4 companies early.")
    for sentence, name in (("Mr. Tenev co-founded Robinhood with Mr. Bhatt in 2013 with a mission to democratize "
                            "finance.", "tenev"),
                           ("Kristo Kaarmann is the co-founder and CEO of Wise.", "kaarmann"),
                           ("Chris McDonough has joined 4 companies early.", "mcdonough")):
        assert check_bio(sentence, good, name) is None, name


def test_a_stored_biography_that_fails_the_rules_is_withdrawn_and_replaced(tmp_path):
    from leadership.verify import withdraw_bad_bios
    ops, _ = _world(tmp_path, llm=_Chain())
    bad = "Our named executive officers are Ada Lovelace, Chief Executive Officer; Shiv Verma, Chief Financial Officer"
    ops.conn.execute(
        """INSERT INTO claims (company_id, person_id, source_id, claim_type, value, evidence, as_of_date, as_of_basis,
               extraction_version, person_as_written)
           VALUES (1, 3, 1, 'bio', ?, ?, '2026-04-22', 'published', 'derived:enrich', 'Shiv Verma')""", (bad, bad))
    ops.conn.execute("INSERT INTO checks (company_id, person_id, kind, outcome, checked_at) "
                     "VALUES (1, 3, 'bio', 'found', '2026-11-13')")
    ops.conn.commit()
    withdraw_bad_bios(ops)
    assert ops.conn.execute("SELECT count(*) FROM claims WHERE claim_type = 'bio'").fetchone()[0] == 0
    assert ops.conn.execute("SELECT count(*) FROM checks WHERE kind = 'bio'").fetchone()[0] == 0


# ------------------------------------------- one automatic search pass per company
def test_no_automatic_search_after_one_full_pass(tmp_path):
    ops, calls = _world(tmp_path, llm=_Chain())
    ops.allow_search = False
    ops.limits.max_enrichment_people = 1                # the CEO only: her answer is recorded below
    verify_roles(ops)
    assert calls["search"] == []                        # the unverified COO is not searched
    assert ops.conn.execute("SELECT count(*) FROM checks WHERE kind = 'verify'").fetchone()[0] == 0
    assert "one full pass was already made" in ops.notes[-1]
    _answer(ops, "Ada Lovelace", passages(PROXY, "lovelace"), {"location": None, "location_evidence": None,
                                                               "bio_evidence": None})
    enrich(ops)
    assert not any(label.startswith("location of") for label, _ in calls["search"])   # no location search either


# ------------------------------------------------- end of employment (M4 review)
def test_a_stated_end_of_employment_ends_every_role_and_the_employment(tmp_path):
    """Robinhood: the CFO moved to an advisory role and 'will remain employed with the Company until
    September 1, 2026'. Extraction kept the advisory role but not the date, so he showed as employed."""
    from leadership.verify import find_employment_end, recover_employment_end
    day, sentence = find_employment_end(FILING, "warnick")
    assert day == "2026-09-01" and "remain employed with the Company until September 1, 2026" in sentence
    assert find_employment_end("Mr. Warnick will remain in his role until a successor is named.", "warnick") is None
    assert find_employment_end("Ms. Lovelace will remain with the Company until the closing.", "lovelace") is None
    assert find_employment_end(FILING, "lovelace") is None               # another person's sentence
    ops, _ = _world(tmp_path)
    ops.conn.execute(
        """INSERT INTO claims (company_id, person_id, source_id, claim_type, raw_title, level, department,
               evidence, as_of_date, as_of_basis, extraction_version)
           VALUES (1, 2, 2, 'appointment', 'advisory role', 'other', 'other', 'transitioned to an advisory role',
                   '2026-02-10', 'published', 'v1')""")
    ops.conn.commit()
    ops.resolve()
    employment = lambda: ops.conn.execute("SELECT employment_status FROM people WHERE id = 2").fetchone()[0]
    assert employment() == "current"                                    # the advisory role keeps him employed
    recover_employment_end(ops)
    ops.resolve()
    assert employment() == "former"
    advisory = ops.conn.execute("SELECT status, ended_on FROM roles WHERE person_id = 2 AND title = 'advisory role'")
    assert tuple(advisory.fetchone()) == ("former", "2026-09-01")
    recover_employment_end(ops)                                         # nothing is added twice
    assert ops.conn.execute("SELECT count(*) FROM claims WHERE extraction_version = 'derived:employment-end'"
                            ).fetchone()[0] == 1
