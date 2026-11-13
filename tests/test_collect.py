"""Collection run on a small fake website served in memory (no network, no keys).

The site is a transport-level fake: the real fetcher, cache, robots handling,
budgets, database writes and run log all run unchanged. No LLM is faked: with
no key configured, the pipeline's own keyword ranking chooses the pages.
"""

import json

import httpx
import pytest

from leadership.collect import classify, collect, headline, print_report
from leadership.config import load_config
from leadership.db import connect
from leadership.sources.site import HomepageError

HOME = """<html><head><title>Acme Robotics | Home</title></head><body>
<a href="/about/leadership">Leadership</a> <a href="/press/acme-names-new-cfo">News</a>
<a href="/private/board-of-directors">Board</a> <a href="/products">Products</a>
<a href="/about/team-archive">Old team page</a> <a href="https://elsewhere.example/x">Partner</a>
</body></html>"""
PAGES = {
    "/": HOME,
    "/robots.txt": "User-agent: *\nDisallow: /private/\nSitemap: https://acme-robotics.com/sitemap.xml\n",
    "/sitemap.xml": "<urlset><url><loc>https://acme-robotics.com/about/leadership</loc></url>"
                    "<url><loc>https://acme-robotics.com/blog/company-update</loc></url></urlset>",
    "/about/leadership": "<html><title>Leadership</title><body><h2>Ada Lovelace</h2>"
                         "<p>Chief Executive Officer</p></body></html>",
    "/press/acme-names-new-cfo": '<html><head><meta property="article:published_time" '
                                 'content="2026-09-15T08:00:00Z"><title>Acme names new CFO</title>'
                                 "</head><body>Grace Hopper will become CFO.</body></html>",
    "/about/team-archive": "<html><title>Team archive</title><body>Earlier team.</body></html>",
    "/blog/company-update": "<html><title>Update</title><body>News from the company.</body></html>",
}


class FakeWeb:
    """Serves the fake site and counts every request that reaches the network layer."""

    def __init__(self, broken=("/about/team-archive",), refused=()):
        self.requests, self.broken, self.refused = [], set(broken), set(refused)
        self.transport = httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        host, path = request.url.host, request.url.path
        if host == "old-acme.com":
            return httpx.Response(301, headers={"location": f"https://acme-robotics.com{path}"})
        if host == "en.wikipedia.org":
            return httpx.Response(200, json={"query": {"search": []}})
        if host == "api.tavily.com":
            return httpx.Response(200, json={"results": [
                {"title": "Acme hires CTO", "url": "https://technews.example/acme-cto",
                 "content": "Acme Robotics (acme-robotics.com) hired a CTO.",
                 "published_date": "Fri, 02 Oct 2026 09:30:00 GMT"},
                {"title": "Acme Robotics Club elects president", "url": "https://school.example/club",
                 "content": "The student robotics club named a new president."},
                {"title": "Profile", "url": "https://www.linkedin.com/in/someone", "content": "x"},
                {"title": "Acme Robotics", "url": "https://tavily.com/knowledge/company/abc",
                 "content": "Acme Robotics (acme-robotics.com) summary."},
                {"title": "Acme Robotics executives", "url": "https://derrick-app.com/company/acme/executives",
                 "content": "Acme Robotics (acme-robotics.com) executives list."},
                {"title": "Acme Robotics people", "url": "https://vectorshift.ai/research/companies/acme/people",
                 "content": "Acme Robotics (acme-robotics.com) leadership and board."},
                {"title": "Acme Robotics executives", "url": "https://exa.ai/websets/directory/acme-executives",
                 "content": "Acme Robotics (acme-robotics.com) executive team, generated."},
                {"title": "About us", "url": "https://the-acme-robotics.io/about",
                 "content": "Acme Robotics toys. Not to be confused with acme-robotics.com."},
                {"title": "Acme in Europe", "url": "https://acme-robotics.eu/team",
                 "content": "European office of acme-robotics.com."}]})
        if host == "the-acme-robotics.io":
            return httpx.Response(200, text="<html><title>About us</title><body>We make toy robots. "
                                            "Our CEO is Someone Else.</body></html>")
        if host == "acme-robotics.eu":
            return httpx.Response(200, text='<html><head><meta property="article:published_time" '
                                            'content="2026-08-01T10:00:00Z"></head>'
                                            '<title>Acme in Europe</title><body>Part of '
                                            '<a href="https://acme-robotics.com/">Acme Robotics</a>.'
                                            "</body></html>")
        if host == "technews.example":
            return httpx.Response(200, text="<html><title>Acme hires CTO</title><body>Story</body></html>")
        if host != "acme-robotics.com":
            return httpx.Response(500)
        if path in self.broken:
            return httpx.Response(500)
        if path in self.refused:
            return httpx.Response(403)
        if path in PAGES:
            return httpx.Response(200, text=PAGES[path])
        return httpx.Response(404)


@pytest.fixture
def no_keys(monkeypatch):
    for name in ("OPENAI_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY", "SEC_USER_AGENT"):
        monkeypatch.delenv(name, raising=False)


def _cfg(**limits):
    cfg = load_config()
    base = {"host_delay_seconds": 0, "max_retries": 1, **limits}
    return cfg.model_copy(update={"limits": cfg.limits.model_copy(update=base)})


def _run(tmp_path, web, cfg=None, url="acme-robotics.com", **kw):
    # These tests cover discovery and fetching; extraction has its own tests.
    kw.setdefault("extract", False)
    return collect(url, cfg or _cfg(), root=tmp_path, transport=web.transport,
                   sleep=lambda s: None, **kw)


def _sources(tmp_path):
    conn = connect(tmp_path / "data" / "leadership.db")
    rows = {r["url"].replace("https://acme-robotics.com", ""): dict(r)
            for r in conn.execute("SELECT * FROM sources")}
    conn.close()
    return rows


def _runs(tmp_path):
    conn = connect(tmp_path / "data" / "leadership.db")
    rows = [dict(r) for r in conn.execute("SELECT * FROM runs ORDER BY id")]
    conn.close()
    return rows


def test_run_discovers_fetches_and_records_sources(tmp_path, no_keys):
    result = _run(tmp_path, FakeWeb(broken=()))
    # The one reason is the board page that robots.txt forbids; it does not change the status.
    assert result.status == "complete" and [r["code"] for r in result.reasons] == ["access_refused"]
    assert result.company == "Acme Robotics"
    src = _sources(tmp_path)
    assert src["/about/leadership"]["source_type"] == "company_site"
    assert src["/about/leadership"]["tier"] == 1 and src["/about/leadership"]["stage"] == "fetched"
    assert len(src["/about/leadership"]["content_hash"]) == 64
    press = src["/press/acme-names-new-cfo"]
    assert (press["source_type"], press["tier"], press["published_at"]) == ("press_release", 2, "2026-09-15")
    assert src["/about/leadership"]["published_at"] is None      # a team page has no date: not guessed
    assert "/products" not in src                                 # not a leadership page
    assert not any("elsewhere.example" in u for u in src)         # other sites are not the company's
    (run,) = _runs(tmp_path)
    assert run["status"] == "complete" and run["finished_at"]
    assert json.loads(run["counters"])["pages"] == len(src)


def test_second_run_makes_no_network_calls(tmp_path, no_keys):
    web = FakeWeb(broken=())
    first = _run(tmp_path, web)
    assert first.network_calls == len(web.requests) > 0
    before = len(web.requests)
    second = _run(tmp_path, web)
    assert len(web.requests) == before and second.network_calls == 0
    assert second.status == "complete"
    assert len(_runs(tmp_path)) == 2                              # a complete run is not resumed


def test_refresh_fetches_again(tmp_path, no_keys):
    web = FakeWeb(broken=())
    _run(tmp_path, web)
    before = len(web.requests)
    _run(tmp_path, web, refresh=True)
    assert len(web.requests) > before


def test_robots_disallowed_page_is_skipped_not_fetched(tmp_path, no_keys):
    web = FakeWeb(broken=())
    result = _run(tmp_path, web)
    board = _sources(tmp_path)["/private/board-of-directors"]
    assert board["stage"] == "skipped" and "robots" in board["last_error"]
    assert not any("/private/" in u for u in web.requests)
    assert result.counters["pages_skipped"] == 1 and result.status == "complete"


def test_page_budget_stops_the_run_and_names_the_limit(tmp_path, no_keys):
    result = _run(tmp_path, FakeWeb(broken=()), _cfg(max_pages=2))
    assert result.status == "budget_exhausted"
    assert result.counters["pages"] == 2 and len(_sources(tmp_path)) == 2
    (reason,) = [r for r in result.reasons if r["code"] == "budget_exhausted"]
    assert reason["detail"] == "max_pages=2 reached" and reason["stage"] == "fetch"


def test_every_condition_is_recorded_not_only_the_headline(tmp_path, no_keys):
    result = _run(tmp_path, FakeWeb(), _cfg(max_pages=5))       # one page fails, and the budget is hit
    codes = {r["code"] for r in result.reasons}
    assert {"partial_failure", "budget_exhausted"} <= codes
    assert result.status == "partial_failure"                    # precedence picks the headline only
    failure = next(r for r in result.reasons if r["code"] == "partial_failure")
    assert failure["url"].endswith("/about/team-archive") and "500" in failure["detail"]
    stored = json.loads(_runs(tmp_path)[0]["reasons"])
    assert stored == result.reasons                              # the database keeps all of them


def test_failed_page_does_not_stop_the_run(tmp_path, no_keys):
    result = _run(tmp_path, FakeWeb())
    src = _sources(tmp_path)
    assert src["/about/team-archive"]["stage"] == "failed"
    assert src["/about/leadership"]["stage"] == "fetched"
    assert result.status == "partial_failure" and result.counters["pages_failed"] == 1


def test_resume_cannot_bypass_the_budget(tmp_path, no_keys):
    web = FakeWeb(broken=())
    first = _run(tmp_path, web, _cfg(max_pages=2))
    again = _run(tmp_path, web, _cfg(max_pages=2))               # same limit: nothing more is fetched
    assert again.run_id == first.run_id and again.status == "budget_exhausted"
    assert again.counters["pages"] == 2 and len(_sources(tmp_path)) == 2
    raised = _run(tmp_path, web, _cfg(max_pages=40))             # limit raised: the run continues
    assert raised.run_id == first.run_id and raised.status == "complete"
    assert raised.counters["pages"] == len(_sources(tmp_path)) > 2
    assert len(_runs(tmp_path)) == 1


def test_interrupted_run_is_detected_and_continued(tmp_path, no_keys):
    web = FakeWeb(broken=())
    conn = connect(tmp_path / "data" / "leadership.db") if (tmp_path / "data").exists() else None
    assert conn is None
    (tmp_path / "data").mkdir()
    conn = connect(tmp_path / "data" / "leadership.db")
    conn.execute("INSERT INTO companies (id, domain, name) VALUES (1, 'acme-robotics.com', 'Acme')")
    conn.execute("INSERT INTO runs (company_id, started_at, status, counters) VALUES "
                 "(1, '2026-11-01T10:00:00', 'running', '{\"pages\": 1}')")
    conn.execute("INSERT INTO sources (company_id, url, source_type, tier, retrieved_at, stage, attempts)"
                 " VALUES (1, 'https://acme-robotics.com/about/leadership', 'company_site', 1,"
                 " '2026-11-01T10:00:05', 'fetched', 1)")
    conn.commit()
    conn.close()
    result = _run(tmp_path, web)
    assert result.run_id == 1 and "resumed (interrupted)" in result.notes[0]
    assert result.status == "complete"
    src = _sources(tmp_path)
    assert src["/about/leadership"]["attempts"] == 1             # already fetched: not done again
    assert src["/about/leadership"]["retrieved_at"] == "2026-11-01T10:00:05"
    assert result.counters["pages"] == len(src)                  # the earlier page still counts
    assert len(_runs(tmp_path)) == 1


def test_failed_source_is_retried_up_to_the_cap(tmp_path, no_keys):
    web = FakeWeb()
    _run(tmp_path, web)
    _run(tmp_path, web)                                          # resumed: second attempt
    assert _sources(tmp_path)["/about/team-archive"]["attempts"] == 2
    hits = sum(u.endswith("/about/team-archive") for u in web.requests)
    third = _run(tmp_path, web)                                  # cap (2) reached: not requested again
    assert sum(u.endswith("/about/team-archive") for u in web.requests) == hits
    assert any("gave up after 2 attempts" in r["detail"] for r in third.reasons)


def test_unreachable_homepage_stops_with_a_clear_error_and_stores_nothing(tmp_path, no_keys):
    with pytest.raises(HomepageError, match="Could not read the homepage of nowhere.example: "
                                            "the site could not be reached"):
        _run(tmp_path, FakeWeb(), url="https://nowhere.example/")
    conn = connect(tmp_path / "data" / "leadership.db")
    assert conn.execute("SELECT count(*) FROM companies").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM sources").fetchone()[0] == 0
    conn.close()


def test_redirected_domain_is_still_the_companys_own_site(tmp_path, no_keys):
    result = _run(tmp_path, FakeWeb(broken=()), url="https://old-acme.com/")
    assert result.status == "complete"
    src = _sources(tmp_path)
    assert src["/about/leadership"]["source_type"] == "company_site"


def test_search_results_are_fetched_cached_and_the_key_stays_out_of_the_log(tmp_path, no_keys, monkeypatch):
    secret = "tvly-TESTSECRETVALUE0123456789"
    monkeypatch.setenv("TAVILY_API_KEY", secret)
    web = FakeWeb(broken=())
    result = _run(tmp_path, web)
    assert result.counters["search_queries"] == 4
    conn = connect(tmp_path / "data" / "leadership.db")
    news = conn.execute("SELECT * FROM sources WHERE url = 'https://technews.example/acme-cto'").fetchone()
    # The search provider said 2026-10-02, but the page states no date: left empty.
    assert (news["source_type"], news["tier"], news["published_at"]) == ("news", 3, None)
    stated = conn.execute("SELECT * FROM sources WHERE url = 'https://acme-robotics.eu/team'").fetchone()
    assert stated["published_at"] == "2026-08-01"       # a date the page itself states is used
    assert conn.execute("SELECT count(*) FROM sources WHERE url LIKE '%linkedin%'").fetchone()[0] == 0
    # A namesake that never mentions the company's domain is not stored as a source.
    assert conn.execute("SELECT count(*) FROM sources WHERE url LIKE '%school.example%'").fetchone()[0] == 0
    conn.close()
    log = result.log_path.read_text(encoding="utf-8")
    events = [json.loads(line) for line in log.splitlines()]
    kinds = {e["event"] for e in events}
    assert {"run_start", "fetch", "search", "run_end"} <= kinds
    assert secret not in log and "Authorization" not in log
    end = events[-1]
    assert end["event"] == "run_end" and end["status"] == "complete"
    assert end["counters"]["search_queries"] == 4 and end["limits"]["search_queries"] == 25
    before = len(web.requests)
    second = _run(tmp_path, web)
    assert len(web.requests) == before and second.counters["search_queries"] == 0


def test_lookalike_site_is_excluded_unless_its_page_names_the_company(tmp_path, no_keys, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-TESTSECRETVALUE0123456789")
    web = FakeWeb(broken=())
    result = _run(tmp_path, web)
    src = _sources(tmp_path)
    namesake = src["https://the-acme-robotics.io/about"]
    assert namesake["stage"] == "skipped" and namesake["content_hash"] is None
    assert namesake["last_error"].startswith("not verified")
    assert "acme-robotics.com" in namesake["last_error"]
    # A second site of the same company proves itself by naming the company's domain.
    assert src["https://acme-robotics.eu/team"]["stage"] == "fetched"
    # An ordinary news site is not a lookalike and is not tested this way.
    assert src["https://technews.example/acme-cto"]["stage"] == "fetched"
    assert result.status == "complete"                  # an excluded namesake is not a failure
    events = [json.loads(line) for line in result.log_path.read_text(encoding="utf-8").splitlines()]
    assert [e["url"] for e in events if e["event"] == "verify"] == ["https://the-acme-robotics.io/about"]
    before = len(web.requests)                          # the exclusion is remembered, not re-fetched
    _run(tmp_path, web)
    assert len(web.requests) == before


def test_generated_summary_pages_of_search_companies_are_never_sources(tmp_path, no_keys, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-TESTSECRETVALUE0123456789")
    web = FakeWeb(broken=())
    _run(tmp_path, web)
    for generated in ("tavily.com/knowledge", "exa.ai/", "vectorshift.ai/", "derrick-app.com/"):   # they name the company, and are still left out
        assert not any(generated in u for u in _sources(tmp_path))
        assert not any(generated in u for u in web.requests)
    assert "https://technews.example/acme-cto" in _sources(tmp_path)    # an original article is kept


def test_page_that_refuses_automated_access_is_skipped_reported_and_not_retried(tmp_path, no_keys, capsys):
    web = FakeWeb(broken=(), refused=("/about/leadership",))
    result = _run(tmp_path, web)
    page = _sources(tmp_path)["/about/leadership"]
    assert page["stage"] == "skipped" and page["content_hash"] is None
    assert page["last_error"] == "the site refused automated access (HTTP 403)"
    # Not a failure: the status is unchanged, but the gap is recorded as its own reason.
    assert result.status == "complete" and result.counters["pages_failed"] == 0
    refused = [r for r in result.reasons if r["code"] == "access_refused"]
    assert {"https://acme-robotics.com/about/leadership"} < {r["url"] for r in refused}
    assert not any(r["code"] == "partial_failure" for r in result.reasons)
    asked = sum(u.endswith("/about/leadership") for u in web.requests)
    second = _run(tmp_path, web)                            # never asked again
    assert sum(u.endswith("/about/leadership") for u in web.requests) == asked == 1
    # The report names the page on every run, including one answered from the cache.
    print_report(second, _cfg(), tmp_path / "data" / "leadership.db")
    out = capsys.readouterr().out
    assert "Pages that could not be read" in out
    assert "https://acme-robotics.com/about/leadership  <- the company's own page or filing" in out
    assert "skipped: the site refused automated access (HTTP 403)" in out


def test_a_server_error_is_still_a_failure_and_is_still_retried(tmp_path, no_keys):
    web = FakeWeb(broken=("/about/leadership",))            # 500, not a refusal
    result = _run(tmp_path, web)
    page = _sources(tmp_path)["/about/leadership"]
    assert page["stage"] == "failed" and result.status == "partial_failure"
    asked = sum(u.endswith("/about/leadership") for u in web.requests)
    _run(tmp_path, web)
    assert sum(u.endswith("/about/leadership") for u in web.requests) > asked


def test_search_query_budget_is_enforced(tmp_path, no_keys, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-TESTSECRETVALUE0123456789")
    web = FakeWeb(broken=())
    result = _run(tmp_path, web, _cfg(max_search_queries=2))
    assert sum("api.tavily.com" in u for u in web.requests) == 2
    assert any(r["detail"] == "max_search_queries=2 reached" for r in result.reasons)


def test_classify_and_headline():
    own = {"acme.com"}
    assert classify("https://www.acme.com/about/team", own) == ("company_site", 1)
    assert classify("https://acme.com/newsroom/new-cfo", own) == ("press_release", 2)
    assert classify("https://acme.com/blog/announcing-x", own) == ("press_release", 2)
    assert classify("https://acme.com/company-news/2026", own) == ("press_release", 2)
    # "press" inside another word is not a newsroom (found on adobe.com/express/...).
    assert classify("https://acme.com/express/print/flyer/promotion-flyer", own) == ("company_site", 1)
    assert classify("https://www.sec.gov/Archives/edgar/data/1/x.htm", own) == ("sec_filing", 1)
    assert classify("https://en.wikipedia.org/wiki/Acme", own) == ("wikipedia", 3)
    assert classify("https://technews.example/acme", own) == ("news", 3)
    assert headline([]) == "complete"
    assert headline([{"code": "insufficient_data"}, {"code": "budget_exhausted"}]) == "budget_exhausted"
    assert headline([{"code": "budget_exhausted"}, {"code": "partial_failure"}]) == "partial_failure"


def test_llm_call_budget_stops_model_use_before_any_call(tmp_path, no_keys, monkeypatch):
    # A key is present, so the model would be used; the limit of 0 must stop it first.
    # No call can succeed with this placeholder key, so a wrongly made call fails the test.
    monkeypatch.setenv("GROQ_API_KEY", "gsk_invalid_key_for_acceptance_test")
    result = _run(tmp_path, FakeWeb(broken=()), _cfg(max_llm_calls=0))
    assert result.status == "budget_exhausted" and result.counters["llm_calls"] == 0
    assert any(r["detail"] == "max_llm_calls=0 reached" for r in result.reasons)
    assert _sources(tmp_path)["/about/leadership"]["stage"] == "fetched"    # keyword ranking still works


def test_search_rounds_limit_of_zero_turns_search_off(tmp_path, no_keys, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-TESTSECRETVALUE0123456789")
    web = FakeWeb(broken=())
    result = _run(tmp_path, web, _cfg(max_search_rounds=0))
    assert not any("api.tavily.com" in u for u in web.requests)
    assert result.counters["search_queries"] == 0 and result.status == "complete"
    assert "web search: skipped, max_search_rounds is 0" in result.notes


def test_run_with_no_readable_company_page_ends_insufficient_data(tmp_path, no_keys):
    refused = ("/about/leadership", "/press/acme-names-new-cfo", "/about/team-archive",
               "/blog/company-update")
    result = _run(tmp_path, FakeWeb(broken=(), refused=refused))
    assert result.status == "insufficient_data"
    (reason,) = [r for r in result.reasons if r["code"] == "insufficient_data"]
    assert reason["detail"] == "no company page other than the homepage could be read"
    assert _runs(tmp_path)[-1]["status"] == "insufficient_data"


def test_every_fetch_and_search_log_entry_has_the_required_fields(tmp_path, no_keys, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-TESTSECRETVALUE0123456789")
    result = _run(tmp_path, FakeWeb(broken=()))
    events = [json.loads(line) for line in result.log_path.read_text(encoding="utf-8").splitlines()]
    fetches = [e for e in events if e["event"] == "fetch"]
    searches = [e for e in events if e["event"] == "search"]
    assert fetches and len(searches) == 4
    for e in events:
        assert e["ts"] and e["event"]
    for e in fetches:
        assert {"url", "status", "cache"} <= e.keys() and e["cache"] in ("hit", "miss")
        if e["cache"] == "miss" and "robots" in (e["error"] or ""):
            assert e["status"] == 0             # never requested, so there is nothing to time
        elif e["cache"] == "miss":
            assert {"retries", "latency_s", "error"} <= e.keys()
    for e in searches:
        assert {"query", "cache", "status", "retries", "results", "latency_s"} <= e.keys()
