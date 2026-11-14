"""The extraction stage inside a collection run. SPEC 3.3, 3.10 to 3.12, 6.3.

Fast tests serve the fake site from test_collect and place recorded model
answers in the extraction cache, so the real stage runs with no model call.
Site links and search results are chosen by the pipeline's own no-model path.
Tests marked `llm` call the real provider.
"""

import hashlib
import json
from datetime import date

import pytest
from test_collect import FakeWeb, _cfg

from leadership import collect as collect_module
from leadership import extract as ex
from leadership.cache import FileCache
from leadership.collect import collect
from leadership.db import connect
from leadership.extract import Target, check_claim, chunks, extraction_version
from leadership.sources.site import describe
from leadership.sources.web import clean_text

TODAY = date(2026, 11, 13)
KEY = "gsk_invalid_key_for_acceptance_test"         # a model is "configured"; no call can succeed
LEADERSHIP = {"about_target_company": True, "page_kind": "official", "claims": [
    {"person": "Ada Lovelace", "organization": "Acme Robotics", "claim_type": "role",
     "raw_title": "Chief Executive Officer", "department": "other",
     "evidence": "Ada Lovelace Chief Executive Officer"}]}
PRESS = {"about_target_company": True, "page_kind": "official", "claims": [
    {"person": "Grace Hopper", "organization": "Acme Robotics", "claim_type": "appointment",
     "raw_title": "CFO", "department": "finance", "effective_pending": True,
     "evidence": "Grace Hopper will become CFO."},
    {"person": "Jane Doe", "organization": "Acme Robotics", "claim_type": "role", "raw_title": "CEO",
     "evidence": "Jane Doe is the chief executive"}]}          # invented: must be rejected
NOTHING = {"about_target_company": True, "page_kind": "official", "claims": []}
ANSWERS = {"/about/leadership": LEADERSHIP, "/press/acme-names-new-cfo": PRESS}


@pytest.fixture
def model_configured(monkeypatch):
    """A key is present, so extraction runs; link and result picking use the no-model path."""
    for name in ("OPENAI_API_KEY", "TAVILY_API_KEY", "SEC_USER_AGENT", "WIKIPEDIA_USER_AGENT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    links, results = collect_module.pick_links, collect_module.pick_results
    monkeypatch.setattr(collect_module, "pick_links", lambda llm, *a, **k: links(None, *a, **k))
    monkeypatch.setattr(collect_module, "pick_results", lambda llm, *a, **k: results(None, *a, **k))


def _run(tmp_path, web, cfg=None, **kw):
    kw.setdefault("verify", False)      # these tests cover extraction; the later stages have their own
    return collect("acme-robotics.com", cfg or _cfg(), root=tmp_path, transport=web.transport,
                   sleep=lambda s: None, today=TODAY, **kw)


def _seed(tmp_path, answers=ANSWERS, default=NOTHING):
    """Record a model answer for every fetched page, exactly where the run will look for it."""
    conn = connect(tmp_path / "data" / "leadership.db")
    pages, llm_cache = FileCache(tmp_path / "cache" / "pages"), FileCache(tmp_path / "cache" / "llm")
    home = conn.execute("SELECT url FROM sources ORDER BY id LIMIT 1").fetchone()["url"]
    target = Target("Acme Robotics", {"acme-robotics.com"}, describe(pages.get(home)["body"]))
    for row in conn.execute("SELECT * FROM sources WHERE stage IN ('fetched', 'extracted')"):
        text = clean_text(pages.get(row["url"])["body"])
        answer = next((a for path, a in answers.items() if row["url"].endswith(path)), default)
        for piece in chunks(text, 3):
            user = ex.user_prompt(target, piece, row["source_type"], row["published_at"])
            digest = hashlib.sha256(user.encode()).hexdigest()
            llm_cache.put(f"extract|{extraction_version('openai/gpt-oss-120b')}|{digest}", answer)
    conn.close()


def _events(result):
    """Events of this run's extraction and later stages.

    Two runs started within the same second write to one log file, so only
    what follows the last `run_start` belongs to the run under test.
    """
    events = [json.loads(line) for line in result.log_path.read_text(encoding="utf-8").splitlines()]
    last = max(i for i, e in enumerate(events) if e["event"] == "run_start")
    return events[last:]


def _prepared(tmp_path, **kw):
    web = FakeWeb(broken=())
    _run(tmp_path, web, extract=False)                  # fetch only, so the pages are known
    _seed(tmp_path, **kw)
    return web


def test_run_extracts_checks_stores_and_resolves(tmp_path, model_configured):
    web = _prepared(tmp_path)
    result = _run(tmp_path, web)
    conn = connect(tmp_path / "data" / "leadership.db")
    roles = {r["person"]: (r["status"], r["canonical_role"], r["status_reason"])
             for r in conn.execute("SELECT * FROM leadership")}
    assert roles["Ada Lovelace"] == ("current", "CEO", "rule4_listed_on_company_page")
    assert roles["Grace Hopper"][:2] == ("announced", "CFO")        # "will become", no date: not current
    assert "Jane Doe" not in roles
    assert [r["person"] for r in conn.execute("SELECT person FROM current_leadership")] == ["Ada Lovelace"]
    # Every source is now extracted, with the version that produced it.
    stages = {r["stage"] for r in conn.execute("SELECT stage FROM sources WHERE stage != 'skipped'")}
    assert stages == {"extracted"}
    assert all(r[0] for r in conn.execute("SELECT extraction_version FROM sources WHERE stage = 'extracted'"))
    # The invented claim was refused and logged with its reason.
    refused = [e for e in _events(result) if e["event"] == "claim_rejected"]
    assert [(e["person"], e["reason"]) for e in refused] == [("Jane Doe", "the evidence quote is not in the document")]
    assert result.counters["claims"] == 2 and result.counters["people"] == 2
    assert result.counters["claims_rejected"] == 1 and result.counters["llm_calls"] == 0
    assert result.status == "complete"
    conn.close()


def test_every_stored_claim_passes_the_evidence_check_again(tmp_path, model_configured):
    web = _prepared(tmp_path)
    _run(tmp_path, web)
    conn = connect(tmp_path / "data" / "leadership.db")
    pages = FileCache(tmp_path / "cache" / "pages")
    target = Target("Acme Robotics", {"acme-robotics.com"})
    rows = conn.execute("SELECT cl.*, p.full_name, s.url FROM claims cl JOIN people p ON p.id = cl.person_id "
                        "JOIN sources s ON s.id = cl.source_id").fetchall()
    assert rows
    for r in rows:
        claim = ex.ExtractedClaim(person=r["person_as_written"], organization="Acme Robotics",
                                  claim_type=r["claim_type"], raw_title=r["raw_title"], evidence=r["evidence"])
        assert check_claim(claim, clean_text(pages.get(r["url"])["body"]), target) is None
    conn.close()


def test_a_second_run_repeats_no_extraction(tmp_path, model_configured):
    web = _prepared(tmp_path)
    _run(tmp_path, web)
    before = len(web.requests)
    again = _run(tmp_path, web)
    events = _events(again)
    assert len(web.requests) == before                              # no fetch repeated
    assert not [e for e in events if e["event"] == "extract"]       # no document touched
    assert not [e for e in events if e["event"] == "llm_call" and e.get("stage") == "extract"]
    assert again.counters["claims"] == 2 and again.status == "complete"


def test_changed_prompt_re_extracts_and_leaves_no_claim_of_the_old_version(tmp_path, model_configured, monkeypatch):
    web = _prepared(tmp_path)
    _run(tmp_path, web)
    conn = connect(tmp_path / "data" / "leadership.db")
    old = {r[0] for r in conn.execute("SELECT DISTINCT extraction_version FROM claims")}
    monkeypatch.setattr(ex, "USER", ex.USER.replace("At most", "No more than"))
    _seed(tmp_path)                                     # the "model" answers the new prompt
    result = _run(tmp_path, web)
    new = {r[0] for r in conn.execute("SELECT DISTINCT extraction_version FROM claims")}
    assert len(old) == 1 and len(new) == 1 and old != new           # nothing stale survives
    assert conn.execute("SELECT count(*) FROM claims").fetchone()[0] == 2
    assert len([e for e in _events(result) if e["event"] == "extract"]) >= 2
    conn.close()


def test_changed_checks_re_process_documents_without_any_model_call(tmp_path, model_configured, monkeypatch):
    web = _prepared(tmp_path)
    _run(tmp_path, web)
    monkeypatch.setattr(collect_module, "code_version", lambda: "newrules")
    monkeypatch.setattr(ex, "code_version", lambda: "newrules")
    result = _run(tmp_path, web)
    events = _events(result)
    assert [e for e in events if e["event"] == "extract"]           # documents were processed again
    assert all(e["cache"] == "hit" for e in events if e["event"] == "llm_call" and e.get("stage") == "extract")
    assert result.counters["llm_calls"] == 0


def test_llm_call_limit_stops_extraction_and_the_run_can_resume(tmp_path, model_configured):
    web = FakeWeb(broken=())
    _run(tmp_path, web, extract=False)                  # nothing recorded: every document needs a call
    result = _run(tmp_path, web, _cfg(max_llm_calls=0))
    assert result.status == "budget_exhausted"
    assert any(r["stage"] == "extract" and r["detail"] == "max_llm_calls=0 reached" for r in result.reasons)
    conn = connect(tmp_path / "data" / "leadership.db")
    assert conn.execute("SELECT count(*) FROM sources WHERE stage = 'extracted'").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM claims").fetchone()[0] == 0
    conn.close()
    # Too little found is not reported as a second problem when the real cause is the limit.
    assert not any(r["code"] == "insufficient_data" and r["stage"] == "extract" for r in result.reasons)
    _seed(tmp_path)                                     # as if the model had answered meanwhile
    resumed = _run(tmp_path, web, _cfg(max_llm_calls=5))            # the limit was raised
    assert resumed.run_id == result.run_id and resumed.status == "complete"
    assert resumed.counters["llm_calls"] == 0                       # cached answers cost nothing
    assert resumed.counters["claims"] == 2


def test_without_a_model_key_the_run_says_no_people_could_be_extracted(tmp_path, monkeypatch):
    for name in ("OPENAI_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY", "SEC_USER_AGENT"):
        monkeypatch.delenv(name, raising=False)
    result = _run(tmp_path, FakeWeb(broken=()))
    assert result.status == "insufficient_data"
    assert any("no LLM key is configured" in r["detail"] for r in result.reasons)


def test_second_search_round_runs_only_when_too_little_was_found(tmp_path, model_configured, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-TESTSECRETVALUE0123456789")
    web = FakeWeb(broken=())
    _run(tmp_path, web, extract=False)
    _seed(tmp_path)                                     # one in-scope person: fewer than three
    result = _run(tmp_path, web)
    assert sum("api.tavily.com" in u for u in web.requests) == 7    # 4 queries in round 1, 3 in round 2
    assert result.counters["search_rounds"] == 2
    note = next(n for n in result.notes if n.startswith("search round 2 ran because"))
    assert "only 1 current leaders found" in note and "No further round is ever run" in note
    ran = [e for e in _events(result) if e["event"] == "search_round"]
    assert len(ran) == 1 and ran[0]["ran"] is True                  # exactly one extra round, never a loop


def test_second_search_round_is_skipped_when_a_ceo_and_three_leaders_are_found(tmp_path, model_configured, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-TESTSECRETVALUE0123456789")
    web = FakeWeb(broken=())
    web.pages = None
    import test_collect
    full = ("<html><title>Leadership</title><body><h2>Ada Lovelace</h2><p>Chief Executive Officer</p>"
            "<h2>Grace Hopper</h2><p>Chief Financial Officer</p><h2>Alan Turing</h2>"
            "<p>Chief Technology Officer</p></body></html>")
    monkeypatch.setitem(test_collect.PAGES, "/about/leadership", full)
    three = {"about_target_company": True, "page_kind": "official", "claims": [
        {"person": n, "organization": "Acme Robotics", "claim_type": "role", "raw_title": t,
         "evidence": f"{n} {t}"} for n, t in [("Ada Lovelace", "Chief Executive Officer"),
                                              ("Grace Hopper", "Chief Financial Officer"),
                                              ("Alan Turing", "Chief Technology Officer")]]}
    _run(tmp_path, web, extract=False)
    _seed(tmp_path, answers={"/about/leadership": three})
    result = _run(tmp_path, web)
    assert sum("api.tavily.com" in u for u in web.requests) == 4    # round 1 only
    assert ("search round 2: not needed (a current CEO and 3 current leaders found; "
            "0 more people are unverified and not counted)") in result.notes
    skipped = [e for e in _events(result) if e["event"] == "search_round"]
    assert len(skipped) == 1 and skipped[0]["ran"] is False


def test_a_third_party_profile_page_cannot_make_a_role_current(tmp_path, model_configured, monkeypatch):
    """R153: decided from what the page is, not from a list of site names."""
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-TESTSECRETVALUE0123456789")
    web = FakeWeb(broken=())
    _run(tmp_path, web, extract=False)
    cto = {"person": "Alan Turing", "organization": "Acme Robotics", "claim_type": "role",
           "raw_title": "Chief Technology Officer", "evidence": "Acme hires CTO"}
    conn = connect(tmp_path / "data" / "leadership.db")
    # Give the news page a publication date, so only its kind decides the outcome.
    conn.execute("UPDATE sources SET published_at = '2026-10-02' WHERE url LIKE '%technews.example%'")
    conn.commit()
    monkeypatch.setattr(FakeWeb, "handle", _with_turing(FakeWeb.handle))
    FileCache(tmp_path / "cache" / "pages").put(
        "https://technews.example/acme-cto",
        {**FileCache(tmp_path / "cache" / "pages").get("https://technews.example/acme-cto"),
         "body": "<html><title>Acme hires CTO</title><body>Acme hires CTO Alan Turing, says the listing."
                 "</body></html>"})
    cto["evidence"] = "Acme hires CTO Alan Turing"
    for kind, expected_tier, expected_status in [("article", 3, "current"),
                                                 ("profile_or_directory", 4, "unverified")]:
        conn.execute("UPDATE sources SET tier = 3, stage = 'fetched' WHERE url LIKE '%technews.example%'")
        conn.commit()
        _seed(tmp_path, answers={"acme-cto": {"about_target_company": True, "page_kind": kind, "claims": [cto]}})
        _run(tmp_path, web)
        tier = conn.execute("SELECT tier FROM sources WHERE url LIKE '%technews.example%'").fetchone()[0]
        status = conn.execute("SELECT status FROM leadership WHERE person = 'Alan Turing'").fetchone()[0]
        assert (tier, status) == (expected_tier, expected_status), kind
    conn.close()


def _with_turing(original):
    return original


# ------------------------------------------------------------------- real model
@pytest.fixture
def real_model(monkeypatch):
    from leadership.config import load_env
    load_env()
    for name in ("TAVILY_API_KEY", "SEC_USER_AGENT", "WIKIPEDIA_USER_AGENT"):
        monkeypatch.delenv(name, raising=False)


def _people(tmp_path):
    conn = connect(tmp_path / "data" / "leadership.db")
    rows = {(r["person"], r["canonical_role"], r["status"]) for r in conn.execute("SELECT * FROM leadership")}
    conn.close()
    return rows


@pytest.mark.llm
def test_real_run_stopped_by_its_limit_resumes_without_repeating_work(tmp_path, real_model):
    web = FakeWeb(broken=())
    first = _run(tmp_path / "a", web, _cfg(max_llm_calls=2))        # link choice, then one document
    assert first.status == "budget_exhausted"
    second = _run(tmp_path / "a", web, _cfg(max_llm_calls=40))
    assert second.run_id == first.run_id and second.status in ("complete", "insufficient_data")
    done: dict[str, int] = {}
    for result in (first, second):
        for e in _events(result):
            if e["event"] == "extract" and "claims" in e:
                done[e["url"]] = done.get(e["url"], 0) + 1
    assert done and all(n == 1 for n in done.values())              # no document extracted twice
    calls = [e for r in (first, second) for e in _events(r)
             if e["event"] == "llm_call" and e.get("stage") == "extract" and e["cache"] == "miss"]
    assert len({e["url"] for e in calls}) == len(calls) or all(e.get("attempt") for e in calls)
    # The result equals that of a run that was never interrupted. That run reuses the same
    # model answers: the test is about resuming, and the model is not fully repeatable.
    import shutil
    shutil.copytree(tmp_path / "a" / "cache" / "llm", tmp_path / "b" / "cache" / "llm")
    _run(tmp_path / "b", FakeWeb(broken=()), _cfg(max_llm_calls=40))
    assert _people(tmp_path / "a") == _people(tmp_path / "b")
    assert ("Ada Lovelace", "CEO", "current") in _people(tmp_path / "a")


@pytest.mark.llm
def test_real_provider_outage_stops_the_run_cleanly_and_keeps_the_data(tmp_path, monkeypatch):
    """R96. A rejected key is the outage: one real request is made and refused."""
    for name in ("OPENAI_API_KEY", "TAVILY_API_KEY", "SEC_USER_AGENT", "WIKIPEDIA_USER_AGENT"):
        monkeypatch.delenv(name, raising=False)
    web = FakeWeb(broken=())
    _run(tmp_path, web, extract=False)                              # fetched with no model involved
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    links = collect_module.pick_links
    monkeypatch.setattr(collect_module, "pick_links", lambda llm, *a, **k: links(None, *a, **k))
    result = _run(tmp_path, web)
    assert result.status == "partial_failure"
    reason = next(r for r in result.reasons if r["stage"] == "extract")
    assert "extraction stopped" in reason["detail"] and "kept" in reason["detail"]
    assert KEY not in result.log_path.read_text(encoding="utf-8")
    conn = connect(tmp_path / "data" / "leadership.db")
    assert conn.execute("SELECT count(*) FROM sources WHERE stage = 'fetched'").fetchone()[0] >= 3
    assert conn.execute("SELECT status FROM runs ORDER BY id DESC LIMIT 1").fetchone()[0] == "partial_failure"
    conn.close()


@pytest.mark.llm
def test_real_output_that_fails_validation_is_retried_once_then_the_document_is_skipped(tmp_path, real_model, monkeypatch):
    """R95. The real model answers; the schema is made impossible to satisfy."""
    from pydantic import model_validator

    class Impossible(ex.Extraction):
        @model_validator(mode="after")
        def _never(self):
            raise ValueError("this test accepts no output")

    monkeypatch.setattr(ex, "Extraction", Impossible)
    web = FakeWeb(broken=())
    _run(tmp_path, web, _cfg(max_pages=2), extract=False)           # the homepage and one page
    result = _run(tmp_path, web, _cfg(max_pages=2))
    events = _events(result)
    tries = [e for e in events if e["event"] == "llm_call" and e.get("stage") == "extract"]
    by_url: dict[str, list] = {}
    for e in tries:
        by_url.setdefault(e["url"], []).append(e["attempt"])
    assert by_url and all(attempts == [1, 2] for attempts in by_url.values())       # one retry each, no more
    assert all(e["valid"] is False and "accepts no output" in e["error"] for e in tries)
    assert result.status == "partial_failure"                       # and the run went on to the end
    skipped = [r for r in result.reasons if "document skipped" in r["detail"]]
    assert len(skipped) == len(by_url) and all(r.get("url") for r in skipped)
    conn = connect(tmp_path / "data" / "leadership.db")
    failed = conn.execute("SELECT last_error FROM sources WHERE stage = 'failed'").fetchall()
    assert len(failed) == len(by_url) and all(r[0].startswith("extraction:") for r in failed)
    assert conn.execute("SELECT count(*) FROM claims").fetchone()[0] == 0
    conn.close()


def test_an_unverified_role_is_never_counted_or_listed_as_a_current_leader(tmp_path, model_configured):
    """A 2020 press release naming a COO is a lead to verify, not a current leader."""
    web = _prepared(tmp_path)
    conn = connect(tmp_path / "data" / "leadership.db")
    conn.execute("UPDATE sources SET published_at = '2020-08-04' WHERE url LIKE '%acme-names-new-cfo'")
    conn.commit()
    old = {"about_target_company": True, "page_kind": "official", "claims": [
        {"person": "Grace Hopper", "organization": "Acme Robotics", "claim_type": "role",
         "raw_title": "CFO", "evidence": "Grace Hopper will become CFO."}]}
    _seed(tmp_path, answers={"/about/leadership": LEADERSHIP, "/press/acme-names-new-cfo": old})
    result = _run(tmp_path, web)
    row = conn.execute("SELECT status, status_reason FROM leadership WHERE person = 'Grace Hopper'").fetchone()
    assert tuple(row) == ("unverified", "rule6_stale_or_weak_evidence")
    current = [r["person"] for r in conn.execute("SELECT person FROM current_leadership")]
    assert current == ["Ada Lovelace"]                              # the view of current leaders excludes her
    assert result.counters["current_leaders"] == 1 and result.counters["unverified_leaders"] == 1
    assert conn.execute("SELECT employment_status FROM people WHERE full_name = 'Grace Hopper'").fetchone()[0] \
        != "current"
    # She does not count toward "enough leaders found", so the second search round is still wanted.
    assert any(n.startswith("search round 2: wanted (only 1 current leaders found)") for n in result.notes)
    conn.close()


# ---------------------------------------------- plan step 1: role words on own domains
def test_role_word_search_runs_once_on_the_companys_own_domains(tmp_path, model_configured, monkeypatch):
    import httpx
    import test_collect
    from leadership.sources.site import describe
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-TESTSECRETVALUE0123456789")
    vp_page = ('<html><head><meta property="article:published_time" content="2026-10-01T09:00:00Z"></head>'
               "<title>New VP</title><body><p>Pooja Anand has been named Vice President, Enterprise Risk at "
               "Acme Robotics. She joins from Globex.</p></body></html>")
    monkeypatch.setitem(test_collect.PAGES, "/press/vp-appointed", vp_page)
    filtered = []
    original = FakeWeb.handle

    def handle(self, request):
        if request.url.host == "api.tavily.com" and json.loads(request.content).get("include_domains"):
            filtered.append(json.loads(request.content))
            return httpx.Response(200, json={"results": [
                {"url": "https://acme-robotics.com/press/vp-appointed", "title": "VP", "content": "x"},
                {"url": "https://acme-robotics.com/press/acme-names-new-cfo", "title": "known", "content": "x"},
                {"url": "https://elsewhere.example/acme-vp", "title": "off the domain", "content": "x"}]})
        return original(self, request)

    monkeypatch.setattr(FakeWeb, "handle", handle)
    web = FakeWeb(broken=())
    _run(tmp_path, web, extract=False)
    _seed(tmp_path)
    home = FileCache(tmp_path / "cache" / "pages").get("https://acme-robotics.com/")["body"]
    target = Target("Acme Robotics", {"acme-robotics.com"}, describe(home))
    vp = {"about_target_company": True, "page_kind": "official", "claims": [
        {"person": "Pooja Anand", "organization": "Acme Robotics", "claim_type": "appointment",
         "raw_title": "Vice President, Enterprise Risk", "department": "other",
         "evidence": "Pooja Anand has been named Vice President, Enterprise Risk at Acme Robotics."}]}
    for piece in chunks(clean_text(vp_page), 3):
        user = ex.user_prompt(target, piece, "press_release", "2026-10-01")
        FileCache(tmp_path / "cache" / "llm").put(
            f"extract|{extraction_version('openai/gpt-oss-120b')}|{hashlib.sha256(user.encode()).hexdigest()}", vp)
    cfg = _cfg(max_enrichment_people=0)
    result = _run(tmp_path, web, cfg, verify=True)
    assert len(filtered) == 3 and all(b["include_domains"] == ["acme-robotics.com"] for b in filtered)
    assert all("acme-robotics.com" not in b["query"] for b in filtered)     # the filter, not the words, limits it
    urls = {r[0] for r in connect(tmp_path / "data" / "leadership.db").execute("SELECT url FROM sources")}
    assert "https://acme-robotics.com/press/vp-appointed" in urls
    assert "https://elsewhere.example/acme-vp" not in urls                  # the provider's filter is not trusted
    conn = connect(tmp_path / "data" / "leadership.db")
    row = conn.execute("SELECT status, level FROM leadership WHERE person = 'Pooja Anand'").fetchone()
    assert tuple(row) == ("current", "vp")                                  # found, checked, decided by the rules
    assert any(n.startswith("role-word search on the company's own domains: 9 results, 1 new pages read")
               for n in result.notes)
    assert conn.execute("SELECT role_search_at FROM companies").fetchone()[0]
    conn.close()
    _run(tmp_path, web, cfg, verify=True)
    assert len(filtered) == 3                                               # once per company: not again


def test_a_pdf_is_recorded_as_not_read_and_never_sent_to_the_model(tmp_path, model_configured):
    web = FakeWeb(broken=())
    _run(tmp_path, web, extract=False)
    _seed(tmp_path)
    pages = FileCache(tmp_path / "cache" / "pages")
    entry = pages.get("https://acme-robotics.com/blog/company-update")
    pages.put("https://acme-robotics.com/blog/company-update",
              {**entry, "content_type": "application/pdf", "body": "%PDF-1.7 binary"})
    result = _run(tmp_path, web)
    conn = connect(tmp_path / "data" / "leadership.db")
    row = conn.execute("SELECT stage, last_error FROM sources WHERE url LIKE '%company-update'").fetchone()
    assert row["stage"] == "skipped" and "application/pdf files are not read" in row["last_error"]
    assert not [e for e in _events(result) if e["event"] == "llm_call" and e.get("url", "").endswith("company-update")]
    conn.close()
