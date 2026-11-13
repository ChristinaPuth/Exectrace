"""Cache, fetcher, link discovery and SEC matching. No network, no keys."""

import json

import httpx
import pytest

from leadership.cache import FileCache
from leadership.config import load_config
from leadership.runlog import RunLog
from leadership.sources import edgar
from leadership.sources.search import round_one_queries
from leadership.sources.site import (lookalike, names_domain, unverified_reason,
    Site, company_name, describe, discover, keep_known, links_in,
                                     mentions_company, normalize_url, pick_results, registrable,
                                     score, shortlist)
from leadership.sources.web import Fetcher, clean_text, content_hash, find_published

LIMITS = load_config().limits.model_copy(update={"host_delay_seconds": 0, "max_retries": 3})


# ----------------------------------------------------------------------- cache
def test_cache_round_trip_and_miss(tmp_path):
    cache = FileCache(tmp_path / "c")
    assert cache.get("https://a.example/") is None
    cache.put("https://a.example/", {"status": 200, "body": "héllo"})
    assert cache.get("https://a.example/") == {"status": 200, "body": "héllo"}
    assert cache.get("https://b.example/") is None


def test_cache_ignores_a_half_written_or_corrupt_entry(tmp_path):
    cache = FileCache(tmp_path / "c")
    cache.put("k", {"v": 1})
    cache.path("k").write_text('{"key": "k", "value": {"v"', encoding="utf-8")    # cut off mid-write
    assert cache.get("k") is None
    cache.put("k", {"v": 2})
    assert cache.get("k") == {"v": 2}
    assert list((tmp_path / "c").glob("*.tmp")) == []           # no temporary files left behind


def test_cache_write_failure_leaves_the_old_entry_intact(tmp_path):
    cache = FileCache(tmp_path / "c")
    cache.put("k", {"v": 1})
    with pytest.raises(TypeError):
        cache.put("k", {"v": object()})                         # cannot be written as JSON
    assert cache.get("k") == {"v": 1}
    assert list((tmp_path / "c").glob("*.tmp")) == []


# --------------------------------------------------------------------- fetcher
def _fetcher(tmp_path, handler, **kw):
    slept = []
    f = Fetcher(LIMITS.model_copy(update=kw.pop("limits", {})), tmp_path / "cache",
                transport=httpx.MockTransport(handler), sleep=slept.append, **kw)
    return f, slept


def test_fetch_is_cached_and_cache_hit_keeps_the_original_retrieval_time(tmp_path):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, text="<html>ok</html>")

    f, _ = _fetcher(tmp_path, handler)
    first = f.get("https://a.example/page", check_robots=False)
    second = f.get("https://a.example/page", check_robots=False)
    assert first.ok and not first.from_cache and second.from_cache
    assert second.body == first.body and second.retrieved_at == first.retrieved_at
    assert len(calls) == 1 and f.network_calls == 1


def test_retries_server_errors_with_backoff_then_succeeds(tmp_path):
    answers = iter([httpx.Response(503), httpx.Response(502), httpx.Response(200, text="ok")])
    f, slept = _fetcher(tmp_path, lambda r: next(answers))
    page = f.get("https://a.example/x", check_robots=False)
    assert page.ok and slept == [1.0, 2.0] and f.network_calls == 3


def test_retry_after_header_is_honored(tmp_path):
    answers = iter([httpx.Response(429, headers={"retry-after": "7"}), httpx.Response(200, text="ok")])
    f, slept = _fetcher(tmp_path, lambda r: next(answers))
    assert f.get("https://a.example/x", check_robots=False).ok and slept == [7.0]


def test_gives_up_after_the_retry_limit_and_does_not_cache_the_failure(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500)

    f, _ = _fetcher(tmp_path, handler, limits={"max_retries": 2})
    page = f.get("https://a.example/x", check_robots=False)
    assert not page.ok and page.error == "HTTP 500" and len(calls) == 3
    f.get("https://a.example/x", check_robots=False)
    assert len(calls) == 6                                      # tried again: a 5xx is not cached


def test_timeout_is_reported_not_raised(tmp_path):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    f, _ = _fetcher(tmp_path, handler, limits={"max_retries": 1})
    page = f.get("https://a.example/x", check_robots=False)
    assert page.status == 0 and "ReadTimeout" in page.error


def test_definite_refusals_are_cached(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(403)

    f, _ = _fetcher(tmp_path, handler)
    assert f.get("https://a.example/x", check_robots=False).error == "HTTP 403"
    assert f.get("https://a.example/x", check_robots=False).from_cache and len(calls) == 1


def test_robots_txt_is_obeyed(tmp_path):
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        return httpx.Response(200, text="page")

    f, _ = _fetcher(tmp_path, handler)
    blocked = f.get("https://a.example/private/team")
    assert blocked.status == 0 and "robots" in blocked.error
    assert f.get("https://a.example/team").ok


def test_site_that_refuses_robots_txt_is_not_crawled(tmp_path):
    def handler(request):
        return httpx.Response(403 if request.url.path == "/robots.txt" else 200, text="x")

    f, _ = _fetcher(tmp_path, handler)
    assert "robots" in f.get("https://closed.example/team").error


def test_missing_robots_allows_and_unreachable_robots_forbids(tmp_path):
    def handler(request):
        if request.url.host == "open.example":
            return httpx.Response(404 if request.url.path == "/robots.txt" else 200, text="x")
        return httpx.Response(503 if request.url.path == "/robots.txt" else 200, text="x")

    f, _ = _fetcher(tmp_path, handler, limits={"max_retries": 0})
    assert f.get("https://open.example/team").ok
    down = f.get("https://down.example/team")
    assert down.status == 0 and "could not be reached" in down.error
    assert "disallowed" not in down.error                        # not blamed on the site's rules


def test_requests_to_one_host_are_spaced_out(tmp_path):
    now = [100.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    f = Fetcher(LIMITS.model_copy(update={"host_delay_seconds": 1.0}), tmp_path / "cache",
                transport=httpx.MockTransport(lambda r: httpx.Response(200, text="x")),
                sleep=sleep, clock=lambda: now[0])
    f.get("https://a.example/1", check_robots=False)
    f.get("https://a.example/2", check_robots=False)
    f.get("https://b.example/1", check_robots=False)             # another host: no wait
    assert slept == [1.0]


def test_fetch_log_has_no_headers(tmp_path):
    log = RunLog(tmp_path / "run.jsonl")
    f = Fetcher(LIMITS, tmp_path / "cache", log,
                transport=httpx.MockTransport(lambda r: httpx.Response(200, text="x")))
    f.get("https://a.example/1", headers={"User-Agent": "Jane Doe jane@example.org"}, check_robots=False)
    text = (tmp_path / "run.jsonl").read_text(encoding="utf-8")
    event = json.loads(text.splitlines()[0])
    assert event["event"] == "fetch" and event["status"] == 200 and event["cache"] == "miss"
    assert "jane@example.org" not in text


# ------------------------------------------------------------------- page text
def test_published_date_is_read_only_when_the_page_states_it():
    assert find_published('<meta property="article:published_time" content="2026-09-15T08:00:00Z">') == "2026-09-15"
    assert find_published('<script type="application/ld+json">{"datePublished": "2025-01-02"}</script>') == "2025-01-02"
    assert find_published('<time datetime="2024-12-31">New Year</time>') == "2024-12-31"
    assert find_published("<html><body>Copyright 2026. Updated last week.</body></html>") is None


def test_content_hash_ignores_markup_changes_but_not_text_changes():
    a = "<html><body><div class='x'><p>Ada Lovelace, Chief Executive Officer</p></div></body></html>"
    b = "<html><body><section id='y'><p>Ada Lovelace, Chief Executive Officer</p></section></body></html>"
    c = a.replace("Executive", "Financial")
    assert content_hash(a) == content_hash(b) != content_hash(c)
    assert "Ada Lovelace" in clean_text(a) and "<p>" not in clean_text(a)


def test_clean_text_drops_scripts_and_styles():
    text = clean_text("<html><style>.a{color:red}</style><script>var secret=1</script>"
                      "<body><h2>Grace Hopper</h2><span>CFO</span></body></html>")
    assert "Grace Hopper" in text and "secret" not in text and "color" not in text


# ------------------------------------------------------------------- discovery
def test_registrable_and_normalize():
    assert registrable("www.robinhood.com") == "robinhood.com"
    assert registrable("investors.robinhood.com") == "robinhood.com"
    assert registrable("www.example.co.uk") == "example.co.uk"
    assert normalize_url("https://a.com/team?utm_source=x#top") == "https://a.com/team"
    assert normalize_url("https://a.com/team?id=3&utm_medium=y") == "https://a.com/team?id=3"


def test_links_are_resolved_and_assets_dropped():
    html = '<a href="/about">A</a><a class="x" href="team#ceo">T</a><a href="/logo.png">L</a>' \
           '<a href="mailto:hi@a.com">M</a><link href="/style.css">'
    assert links_in(html, "https://a.com/company/") == ["https://a.com/about", "https://a.com/company/team"]


def test_company_name_from_site_metadata():
    assert company_name('<meta property="og:site_name" content="Acme Robotics">', "acme.com") == "Acme Robotics"
    assert company_name("<title>Investing for everyone | Robinhood</title>", "robinhood.com") == "Robinhood"
    assert company_name("<title>Welcome</title>", "campfire.ai") == "Campfire"
    assert company_name("<title>Acme Robotics | Home</title>", "acme-robotics.com") == "Acme Robotics"
    assert company_name("<title>Home</title>", "acme-robotics.com") == "Acme Robotics"


def test_ranking_prefers_leadership_pages_and_executive_news():
    urls = ["https://a.com/pricing", "https://a.com/about/leadership", "https://a.com/about/legal/terms/x",
            "https://a.com/newsroom/a-com-welcomes-new-chief-technology-officer",
            "https://a.com/newsroom/five-tips-for-saving", "https://a.com/blog"]
    ranked = shortlist(urls, 10)
    assert ranked[0] == "https://a.com/newsroom/a-com-welcomes-new-chief-technology-officer"
    assert ranked[1] == "https://a.com/about/leadership"
    assert "https://a.com/pricing" not in ranked and score("https://a.com/pricing") == 0
    assert shortlist(urls, 2) == ranked[:2]


def test_link_picker_output_cannot_add_urls_that_were_not_offered():
    offered = ["https://a.com/team", "https://a.com/press"]
    chosen = ["https://a.com/team", "https://evil.example/steal", "https://a.com/team",
              "https://a.com/TEAM", None, 7, "https://a.com/press"]
    assert keep_known(chosen, offered) == ["https://a.com/team", "https://a.com/press"]


def test_sitemap_index_is_followed_within_the_limit(tmp_path):
    fetched = []
    pages = {
        "/": '<html><title>Acme</title><a href="/pricing">P</a></html>',
        "/robots.txt": "Sitemap: https://acme.com/sitemap.xml\n",
        "/sitemap.xml": "<sitemapindex>" + "".join(
            f"<sitemap><loc>https://acme.com/sitemap-{n}.xml</loc></sitemap>"
            for n in ("stocks", "newsroom", "support", "marketing", "learn")) + "</sitemapindex>",
        "/sitemap-newsroom.xml": "<urlset><url><loc>https://acme.com/newsroom/new-cfo</loc></url></urlset>",
        "/sitemap-marketing.xml": "<urlset><url><loc>https://acme.com/about-us</loc></url>"
                                  "<url><loc>https://other.com/about</loc></url></urlset>",
    }

    def handler(request):
        fetched.append(request.url.path)
        return httpx.Response(200, text=pages.get(request.url.path, "<urlset></urlset>"))

    f, _ = _fetcher(tmp_path, handler)
    site = discover(f, "acme.com", max_sitemaps=3)               # index + the two relevant files
    assert "https://acme.com/newsroom/new-cfo" in site.candidates
    assert "https://acme.com/about-us" in site.candidates
    assert "https://other.com/about" not in site.candidates      # another company's page
    assert "/sitemap-stocks.xml" not in fetched and "/sitemap-support.xml" not in fetched


# ---------------------------------------------------------------------- search

def test_queries_are_anchored_to_the_company_domain():
    queries = round_one_queries("Campfire", "campfire.ai")
    assert len(queries) == 4 and all('"Campfire" campfire.ai' in q for q in queries)


def test_namesake_results_are_dropped_without_an_llm(tmp_path):
    site = Site(domain="meetcampfire.com", home_url="https://campfire.ai/", start_url="https://meetcampfire.com/",
                name="Campfire", domains={"meetcampfire.com", "campfire.ai"})
    results = [
        {"url": "https://campfire.org/leadership", "title": "Camp Fire leadership", "content": "Youth nonprofit."},
        {"url": "https://news.example/a", "title": "Campfire CEO interview",
         "content": "The founder of campfire.ai on accounting."},
        {"url": "https://campfire.ai/blog/series-b", "title": "Series B", "content": "Funding."},
    ]
    kept, calls = pick_results(None, site, results, 10, FileCache(tmp_path / "llm"), RunLog(None))
    assert kept == ["https://campfire.ai/blog/series-b", "https://news.example/a"] and calls == 0
    assert not mentions_company(results[0], site.domains)        # campfire.org is another organisation


def test_homepage_description_is_read_from_metadata():
    assert describe('<meta name="description" content="AI-native accounting software for finance teams">')         == "AI-native accounting software for finance teams"
    assert describe("<html><body>No metadata</body></html>") == ""


# ------------------------------------------------------------------------- SEC
TICKERS = {"0": {"cik_str": 1783879, "ticker": "HOOD", "title": "Robinhood Markets, Inc."},
           "1": {"cik_str": 111, "ticker": "CAMP", "title": "Campfire Foods Holdings Corp"},
           "2": {"cik_str": 222, "ticker": "ACM", "title": "Acme Corp"},
           "3": {"cik_str": 333, "ticker": "ACME", "title": "ACME Inc."}}


def test_registrant_is_matched_only_on_an_exact_core_name():
    assert edgar.match_registrant(TICKERS, "Robinhood") == (1783879, "Robinhood Markets, Inc.")
    assert edgar.match_registrant(TICKERS, "Campfire") is None       # "Campfire Foods" is another company
    assert edgar.match_registrant(TICKERS, "Acme") is None           # two registrants: ambiguous
    assert edgar.match_registrant(TICKERS, "Unlisted Startup") is None


def test_filings_selected_are_latest_annual_proxy_and_officer_change_8ks():
    recent = {"form": ["8-K", "8-K", "10-Q", "DEF 14A", "8-K", "10-K", "8-K", "8-K", "10-K"],
              "filingDate": [f"2026-0{9 - i}-01" for i in range(9)],
              "accessionNumber": [f"0001783879-26-00000{i}" for i in range(9)],
              "primaryDocument": [f"doc{i}.htm" for i in range(9)],
              "items": ["2.02,9.01", "5.02", "", "", "5.02,9.01", "", "5.02", "5.02", ""]}
    picked = edgar.select_filings({"cik": "0001783879", "filings": {"recent": recent}}, max_8k=2)
    assert [(f.form, f.filed) for f in picked] == [
        ("8-K", "2026-08-01"), ("DEF 14A", "2026-06-01"), ("8-K", "2026-05-01"), ("10-K", "2026-04-01")]
    assert picked[0].url == "https://www.sec.gov/Archives/edgar/data/1783879/000178387926000001/doc1.htm"


def test_sec_lookup_is_skipped_without_a_contact_and_rejects_a_different_website(tmp_path, monkeypatch):
    def handler(request):
        if "company_tickers" in request.url.path:
            return httpx.Response(200, json=TICKERS)
        if "submissions" in request.url.path:
            return httpx.Response(200, json={"cik": "1783879", "website": "https://www.other-firm.com",
                                             "filings": {"recent": {}}})
        return httpx.Response(404)

    f, _ = _fetcher(tmp_path, handler)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    assert edgar.find_filings(f, "Robinhood", {"robinhood.com"}, 3)[1].startswith("skipped")
    assert f.network_calls == 0
    monkeypatch.setenv("SEC_USER_AGENT", "Test Runner test@example.org")
    filings, note = edgar.find_filings(f, "Robinhood", {"robinhood.com"}, 3)
    assert filings == [] and note.startswith("rejected")


# ------------------------------------------------------------ real model calls
@pytest.mark.llm
def test_real_model_picks_leadership_pages_and_only_offered_urls(tmp_path):
    from leadership.config import load_env
    from leadership.llm import LLM
    from leadership.sources.site import pick_links
    load_env()
    offered = ["https://acme.com/pricing", "https://acme.com/about/leadership",
               "https://acme.com/newsroom/acme-welcomes-jane-doe-as-chief-financial-officer",
               "https://acme.com/blog/five-tips-for-saving", "https://acme.com/careers",
               "https://acme.com/legal/terms"]
    cache, log = FileCache(tmp_path / "llm"), RunLog(tmp_path / "run.jsonl")
    picked, calls = pick_links(LLM(load_config(), "extract"), "Acme", offered, 3, cache, log)
    assert 1 <= calls <= 2 and 1 <= len(picked) <= 3
    assert set(picked) <= set(offered)                           # nothing invented
    assert "https://acme.com/about/leadership" in picked
    assert "https://acme.com/legal/terms" not in picked
    again, calls = pick_links(LLM(load_config(), "extract"), "Acme", offered, 3, cache, log)
    assert again == picked and calls == 0                        # second time from the cache


@pytest.mark.llm
def test_real_model_drops_a_namesake_search_result(tmp_path):
    from leadership.config import load_env
    from leadership.llm import LLM
    load_env()
    site = Site(domain="campfire.ai", home_url="https://campfire.ai/", start_url="https://campfire.ai/",
                name="Campfire", domains={"campfire.ai"},
                description="AI-native ERP and accounting software for finance teams")
    results = [
        {"url": "https://campfire.org/leadership", "title": "Leadership | Camp Fire",
         "content": "Camp Fire is a national youth development nonprofit. Meet our CEO and board."},
        {"url": "https://www.cfodive.com/news/campfire-ceo-accounting", "title": "Campfire CEO on AI accounting",
         "content": "The CEO of accounting software startup Campfire discusses its ERP product."},
    ]
    kept, _ = pick_results(LLM(load_config(), "extract"), site, results, 10,
                           FileCache(tmp_path / "llm"), RunLog(tmp_path / "run.jsonl"))
    assert kept == ["https://www.cfodive.com/news/campfire-ceo-accounting"]


def _campfire():
    return Site(domain="meetcampfire.com", home_url="https://meetcampfire.com/",
                start_url="https://campfire.ai/", name="Campfire",
                domains={"meetcampfire.com", "campfire.ai"}, candidates=[], description="")


def test_a_domain_is_named_only_as_a_whole_name():
    own = {"campfire.ai"}
    assert names_domain("see https://www.campfire.ai/blog", own)
    assert names_domain("Campfire (CAMPFIRE.AI) raised", own)
    assert not names_domain("https://www.thecampfire.ai/about", own)
    assert not names_domain("my-campfire.ai and campfire.aid", own)


def test_lookalike_sites_are_recognised():
    site = _campfire()
    assert lookalike("https://www.thecampfire.ai/about", site)
    assert lookalike("https://campfire-labs.com/", site)
    assert not lookalike("https://campfire.ai/blog", site)              # the company's own
    assert not lookalike("https://blog.meetcampfire.com/x", site)
    assert not lookalike("https://www.cfodive.com/news/campfire-ceo", site)   # name only in the path


def test_only_lookalikes_need_proof_and_the_proof_is_the_companys_domain():
    site = _campfire()
    other = "<html><body>Campfire AI is a Brussels-based conversational AI agency.</body></html>"
    reason = unverified_reason("https://www.thecampfire.ai/about", other, site)
    assert reason and "campfire.ai or meetcampfire.com" in reason and "  " not in reason
    same = '<html><body>Now part of <a href="https://campfire.ai">Campfire</a></body></html>'
    assert unverified_reason("https://www.thecampfire.ai/about", same, site) is None
    assert unverified_reason("https://www.cfodive.com/news/x", other, site) is None


def test_product_pages_with_press_inside_a_word_are_not_ranked_as_news():
    # Real case: adobe.com/express/... template pages were ranked as executive news.
    assert score("https://www.adobe.com/express/print/flyer/coffee-shop-promotion-flyer") == 0
    assert score("https://www.adobe.com/express/discover/messages/welcome") == 0
    assert score("https://acme.com/press/acme-names-new-cfo") > 0
    assert score("https://acme.com/newsroom/acme-welcomes-chief-officer") > 0


class _AnsweredEmpty:
    """A cache that already holds the model's answer "none of these" for any question."""

    def get(self, key):
        return {"urls": []}

    def put(self, key, value):
        raise AssertionError("nothing new should be cached")


def test_valid_empty_link_selection_is_respected_not_treated_as_a_failure(tmp_path, monkeypatch):
    # Real case: for adobe.com the model chose none of 120 offered links, and the old
    # code called that a failure and fetched the 12 best keyword matches instead.
    from leadership.llm import LLM
    from leadership.sources.site import pick_links
    monkeypatch.setenv("GROQ_API_KEY", "gsk_invalid_key_for_acceptance_test")    # never used: cache hit
    offered = ["https://acme.com/express/print/flyer/coffee-shop-promotion-flyer",
               "https://acme.com/express/print/card/blue-president-day-card"]
    problems: list[str] = []
    log = RunLog(tmp_path / "run.jsonl")
    picked, calls = pick_links(LLM(load_config(), "extract"), "Acme", offered, 12,
                               _AnsweredEmpty(), log, problems)
    assert picked == [] and calls == 0
    assert problems == []                                   # so no partial_failure reason
    events = [json.loads(line) for line in (tmp_path / "run.jsonl").read_text(encoding="utf-8").splitlines()]
    pick = [e for e in events if e["event"] == "link_pick"][-1]
    assert pick["method"] == "llm" and pick["chosen"] == 0


def test_without_a_model_the_keyword_ranking_is_still_used(tmp_path):
    from leadership.sources.site import pick_links
    offered = ["https://acme.com/about/leadership", "https://acme.com/press/new-cfo"]
    picked, calls = pick_links(None, "Acme", offered, 12, FileCache(tmp_path / "llm"),
                               RunLog(tmp_path / "run.jsonl"))
    assert picked == offered and calls == 0


@pytest.mark.llm
def test_real_model_may_choose_none_and_that_is_not_a_failure(tmp_path):
    from leadership.config import load_env
    from leadership.llm import LLM
    from leadership.sources.site import pick_links
    load_env()
    offered = ["https://acme.com/express/print/flyer/coffee-shop-promotion-flyer",
               "https://acme.com/express/print/card/green-and-blue-welcome-card",
               "https://acme.com/express/create/gif/welcome"]
    problems: list[str] = []
    picked, calls = pick_links(LLM(load_config(), "extract"), "Acme", offered, 12,
                               FileCache(tmp_path / "llm"), RunLog(tmp_path / "run.jsonl"), problems)
    assert picked == [] and problems == [] and 1 <= calls <= 2
