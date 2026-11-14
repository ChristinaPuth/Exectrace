"""One collection run: discover sources, fetch them, extract claims, rebuild roles.

SPEC sections 3.1 to 3.6, 3.10 and 3.11. This is a fixed workflow, not an
agent. The model picks links, drops namesake search results and extracts
claims; it never fetches, never decides a status and never chooses the next step.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .cache import FileCache
from .config import ROOT, Config, ConfigError, configured_providers
from .db import connect, drop_people_without_claims, rebuild_roles, store_document
from .extract import (Malformed, OverBudget, ProviderDown, Target, code_version, extract_document,
                      processing_version)
from .llm import LLM
from .runlog import RunLog
from .verify import Ops, enrich, recover_dates, recover_earlier_titles, recover_employment_end, verify_roles
from .sources import edgar, wikipedia
from .sources.search import Search, SearchError, role_word_queries, round_one_queries, round_two_queries
from .sources.site import (NEWS_PATH, HomepageError, discover, pick_links, pick_results, registrable,
                           unverified_reason,
                           shortlist)
from .sources.web import Fetcher, clean_text, content_hash, find_published, find_title

RESUMABLE = ("interrupted", "partial_failure", "budget_exhausted")
PRECEDENCE = ("partial_failure", "budget_exhausted", "insufficient_data")
_NO_FETCH = {"linkedin.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
             "youtube.com", "tiktok.com",       # login walls or personal profiles
             # Search companies' machine-generated summaries: not original sources, so
             # never evidence for a claim.
             "tavily.com", "exa.ai", "vectorshift.ai", "derrick-app.com"}


@dataclass
class Planned:
    url: str
    source_type: str
    tier: int
    published_at: str | None = None     # known before fetching: an SEC filing date only
    headers: dict | None = None
    from_search: bool = False           # found by web search: checked after fetching


@dataclass
class RunResult:
    run_id: int
    company: str
    status: str
    reasons: list[dict]
    counters: dict
    network_calls: int
    log_path: Path | None
    notes: list[str] = field(default_factory=list)


def classify(url: str, company_domains: set[str]) -> tuple[str, int]:
    """Source type and reliability tier from the URL alone. SPEC section 3.3."""
    parts = urlsplit(url)
    host = parts.netloc.lower()
    if host.endswith("sec.gov"):
        return "sec_filing", 1
    if host.endswith("wikipedia.org"):
        return "wikipedia", 3
    if registrable(host) in company_domains:
        return ("press_release", 2) if NEWS_PATH.search(parts.path) else ("company_site", 1)
    return "news", 3


class Budget:
    """Usage counters checked against config limits. Cumulative across resumes."""

    LIMITS = {"pages": "max_pages", "search_queries": "max_search_queries",
              "llm_calls": "max_llm_calls"}

    def __init__(self, limits, counters: dict, reasons: list[dict]):
        self.limits, self.c, self.reasons = limits, counters, reasons
        for name in ("pages", "pages_failed", "pages_skipped", "search_queries", "llm_calls",
                     "sources_found"):
            self.c.setdefault(name, 0)

    def take(self, name: str, stage: str) -> bool:
        limit = getattr(self.limits, self.LIMITS[name])
        if self.c[name] >= limit:
            add_reason(self.reasons, "budget_exhausted", stage,
                       f"{self.LIMITS[name]}={limit} reached", limit=self.LIMITS[name])
            return False
        self.c[name] += 1
        return True


def add_reason(reasons: list[dict], code: str, stage: str, detail: str, **extra) -> None:
    """Record a condition once per (code, stage, detail). Every condition is kept."""
    entry = {"code": code, "stage": stage, "detail": detail, **extra}
    if entry not in reasons:
        reasons.append(entry)


def headline(reasons: list[dict]) -> str:
    codes = {r["code"] for r in reasons}
    return next((s for s in PRECEDENCE if s in codes), "complete")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _open_run(conn: sqlite3.Connection, company_id: int) -> tuple[int, dict, list[dict], str]:
    """Continue a resumable run, or start a new one. Returns (id, counters, reasons, mode)."""
    last = conn.execute("SELECT * FROM runs WHERE company_id = ? ORDER BY id DESC LIMIT 1",
                        (company_id,)).fetchone()
    if last and last["status"] == "running":        # the previous process never finished
        reasons = json.loads(last["reasons"])
        add_reason(reasons, "interrupted", "run", "the previous process stopped before finishing")
        conn.execute("UPDATE runs SET status = 'interrupted', reasons = ? WHERE id = ?",
                     (json.dumps(reasons), last["id"]))
        conn.commit()
        last = conn.execute("SELECT * FROM runs WHERE id = ?", (last["id"],)).fetchone()
    if last and last["status"] in RESUMABLE:
        # Counters carry over, so a restart cannot bypass a budget. Old reasons
        # are kept in the log; the list is rebuilt from what is true this time.
        conn.execute("UPDATE runs SET status = 'running', finished_at = NULL WHERE id = ?",
                     (last["id"],))
        conn.commit()
        return last["id"], json.loads(last["counters"]), [], f"resumed ({last['status']})"
    cur = conn.execute("INSERT INTO runs (company_id, started_at) VALUES (?, ?)",
                       (company_id, _now()))
    conn.commit()
    return cur.lastrowid, {}, [], "new"


IN_SCOPE = ("c_level", "vp", "head")
ROLE_SEARCH_PAGES = 6       # new own-domain pages read by the role-word search


def collect(url: str, cfg: Config, *, refresh: bool = False, root: Path = ROOT,
            transport: httpx.BaseTransport | None = None, sleep=None,
            extract: bool = True, verify: bool = True, today: date | None = None) -> RunResult:
    """Run one company: discover, fetch, extract, resolve, then verify and enrich.

    `extract=False` stops after fetching; `verify=False` stops after the first resolution.
    """
    domain = urlsplit(url if "://" in url else "https://" + url).netloc.lower().removeprefix("www.")
    if not domain or "." not in domain:
        raise ConfigError(f"'{url}' is not a company domain or URL.")
    conn = connect(_db_path(root))
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    log = RunLog(root / "logs" / f"collect-{domain}-{stamp}.jsonl")
    kw = {"sleep": sleep} if sleep else {}
    fetcher = Fetcher(cfg.limits, root / "cache", log, refresh=refresh, transport=transport, **kw)
    search = Search(cfg.limits, root / "cache", log, refresh=refresh, transport=transport,
                    max_results=cfg.discovery.search_results_per_query, **kw)
    notes: list[str] = []
    try:
        # ------------------------------------------------------------ homepage
        try:
            site = discover(fetcher, url, max_sitemaps=cfg.discovery.max_sitemaps)
        except HomepageError as e:
            log.event("run_end", status="aborted", error=str(e))
            raise
        conn.execute("INSERT INTO companies (domain, name) VALUES (?, ?) "
                     "ON CONFLICT(domain) DO UPDATE SET name = excluded.name", (domain, site.name))
        company_id = conn.execute("SELECT id FROM companies WHERE domain = ?", (domain,)).fetchone()[0]
        run_id, counters, reasons, mode = _open_run(conn, company_id)
        budget = Budget(cfg.limits, counters, reasons)
        log.event("run_start", run_id=run_id, domain=domain, company=site.name, mode=mode,
                  home_url=site.home_url, refresh=refresh)
        notes.append(f"run {run_id} ({mode}); company name from the site: {site.name}")

        def save_run(status: str = "running") -> None:
            conn.execute("UPDATE runs SET status = ?, reasons = ?, counters = ?, finished_at = ? "
                         "WHERE id = ?", (status, json.dumps(reasons), json.dumps(counters),
                                          None if status == "running" else _now(), run_id))
            conn.commit()

        # ------------------------------------------------------------ discover
        plan: list[Planned] = [Planned(site.start_url, "company_site", 1)]
        problems: list[str] = []            # model failures that forced a fallback

        offered = shortlist(site.candidates, cfg.discovery.link_candidates)
        llm = LLM(cfg, "extract") if configured_providers() else None
        if llm is None:
            notes.append("no LLM key: site pages chosen by keyword ranking")
        if llm is None or budget.take("llm_calls", "discover"):
            picked, calls = pick_links(llm, site.name, offered, cfg.discovery.site_pages,
                                       FileCache(root / "cache" / "llm"), log, problems)
            if llm is not None:
                counters["llm_calls"] += calls - 1  # cache hits do not count; a retry does
        else:
            picked = offered[:cfg.discovery.site_pages]
        plan += [Planned(u, *classify(u, site.domains)) for u in picked]
        notes.append(f"company site: {len(site.candidates)} links found, {len(offered)} shortlisted, "
                     f"{len(picked)} chosen")

        filings, note = edgar.find_filings(fetcher, site.name, site.domains, cfg.discovery.filings_8k)
        notes.append("SEC EDGAR: " + note)
        log.event("discover", source="edgar", note=note)
        if note.startswith("failed"):
            add_reason(reasons, "partial_failure", "discover", "SEC EDGAR: " + note)
        sec_headers = {"User-Agent": edgar.user_agent()}
        plan += [Planned(f.url, "sec_filing", 1, f.filed, sec_headers) for f in filings]

        wiki_url, wiki_headers, note = wikipedia.find_article(fetcher, site.name, site.domains)
        notes.append("Wikipedia: " + note)
        log.event("discover", source="wikipedia", note=note)
        if wiki_url:
            plan.append(Planned(wiki_url, "wikipedia", 3, headers=wiki_headers))

        home_domain = registrable(urlsplit(site.home_url).netloc)

        def search_for(queries: list[str], label: str, limit: int | None = None) -> list[Planned]:
            results = []
            for q in queries:
                cached = search.cache.get(f"tavily|{search.max_results}|{q}") and not refresh
                if not cached and not budget.take("search_queries", "discover"):
                    break
                try:
                    found, _ = search.query(q)
                except SearchError as e:
                    add_reason(reasons, "partial_failure", "discover", f"web search failed: {e}", query=q)
                    break
                results += found
            planned_urls = {p.url for p in plan}
            usable = [r for r in results if r["url"] not in planned_urls
                      and registrable(urlsplit(r["url"]).netloc) not in _NO_FETCH]
            # Namesakes are common ("Campfire"): keep only results about this company.
            if llm is None or budget.take("llm_calls", "discover"):
                relevant, calls = pick_results(llm, site, usable, limit or cfg.discovery.news_pages,
                                               FileCache(root / "cache" / "llm"), log, problems)
                if llm is not None:
                    counters["llm_calls"] += calls - 1
            else:
                relevant = []
            notes.append(f"{label}: {len(results)} results, {len(usable)} usable, "
                         f"{len(relevant)} about this company and added")
            # The search provider's date is not used: for a profile page it is often
            # not a publication date at all. Only a date the page itself states counts.
            return [Planned(u, *classify(u, site.domains), from_search=True) for u in relevant]

        if not search.available:
            notes.append("web search: skipped, TAVILY_API_KEY is not set")
        elif cfg.limits.max_search_rounds < 1:
            notes.append("web search: skipped, max_search_rounds is 0")
        else:
            plan += search_for(round_one_queries(site.name, home_domain), "web search")
            counters["search_rounds"] = max(counters.get("search_rounds", 0), 1)
        for problem in problems:
            add_reason(reasons, "partial_failure", "discover", problem)
            notes.append("degraded: " + problem)
        reported = len(problems)
        counters["sources_found"] = len(plan)
        save_run()

        # --------------------------------------------------------------- fetch
        def fetch(planned: list[Planned]) -> None:
            for p in planned:
                row = conn.execute("SELECT * FROM sources WHERE company_id = ? AND url = ?",
                                   (company_id, p.url)).fetchone()
                if row and not refresh and row["stage"] in ("fetched", "extracted", "skipped"):
                    continue                            # done in an earlier run or before a resume
                if row and row["stage"] == "failed" and row["attempts"] >= cfg.limits.max_source_attempts:
                    add_reason(reasons, "partial_failure", "fetch",
                               f"gave up after {row['attempts']} attempts: {row['last_error']}", url=p.url)
                    continue
                if not budget.take("pages", "fetch"):
                    break
                page = fetcher.get(p.url, headers=p.headers)
                if page.ok:
                    stage, error = "fetched", None
                    published = p.published_at
                    if not published and p.source_type in ("press_release", "news"):
                        published = find_published(page.body)   # a page's own stated date only
                    digest, title = content_hash(page.body), find_title(page.body)
                    doubt = unverified_reason(p.url, page.body, site) if p.from_search else None
                    if doubt:                           # a namesake's site: kept as a record, never used
                        stage, error, digest = "skipped", doubt, None
                        counters["pages_skipped"] += 1
                        log.event("verify", url=p.url, kept=False, reason=doubt)
                else:
                    published, digest, title = p.published_at, None, None
                    error = page.error
                    refused = page.status in (401, 403)
                    if refused:
                        error = f"the site refused automated access (HTTP {page.status})"
                    if refused or (error and "robots" in error):
                        # A refusal is the site's decision, not a fault: never retried, never
                        # bypassed. For the company's own pages and filings it is recorded as
                        # a reason too, so the gap is visible; it does not change the status.
                        stage = "skipped"
                        counters["pages_skipped"] += 1
                        if p.tier <= 2:
                            add_reason(reasons, "access_refused", "fetch", error, url=p.url)
                    else:
                        stage = "failed"
                        counters["pages_failed"] += 1
                        add_reason(reasons, "partial_failure", "fetch", error, url=p.url)
                # Saved immediately, one transaction per source, so a crash loses nothing.
                conn.execute(
                    """INSERT INTO sources (company_id, url, source_type, tier, title, published_at,
                           retrieved_at, content_hash, stage, attempts, last_error)
                       VALUES (?,?,?,?,?,?,?,?,?,1,?)
                       ON CONFLICT(company_id, url) DO UPDATE SET
                           title = excluded.title, published_at = excluded.published_at,
                           retrieved_at = excluded.retrieved_at, content_hash = excluded.content_hash,
                           stage = excluded.stage, attempts = sources.attempts + 1,
                           last_error = excluded.last_error""",
                    (company_id, p.url, p.source_type, p.tier, title, published,
                     page.retrieved_at or _now(), digest, stage, error))
                save_run()

        fetch(plan)

        # ------------------------------------------------------------- extract
        today = today or date.today()
        stopped: list[str] = []             # why extraction ended early, if it did

        def extract_pending() -> None:
            """Extract every fetched document not yet extracted by the current prompt and model."""
            versions = {processing_version(model) for _, model in llm.chain}
            target = Target(site.name, set(site.domains), site.description)
            rows = conn.execute("SELECT * FROM sources WHERE company_id = ? AND stage IN "
                                "('fetched', 'extracted') ORDER BY tier, id", (company_id,)).fetchall()
            for row in rows:
                cached = fetcher.cache.get(row["url"])
                kind = ((cached or {}).get("content_type") or "").lower()
                binary = bool(kind) and "html" not in kind and not kind.startswith("text/")
                if row["stage"] == "extracted" and row["extraction_version"] in versions and not binary:
                    continue                        # done; a resume never repeats it
                if not cached or not cached.get("body"):
                    log.event("extract", url=row["url"], skipped="the page is not in the cache")
                    continue
                if binary:
                    # A PDF or other binary file: its bytes are not text, and sending them to
                    # the model wastes calls. Reading PDFs is not built (discovery plan, step 3).
                    reason = f"not read: {kind.split(';')[0]} files are not read (only web pages are)"
                    with conn:
                        conn.execute("DELETE FROM claims WHERE source_id = ?", (row["id"],))
                        conn.execute("UPDATE sources SET stage = 'skipped', last_error = ? WHERE id = ?",
                                     (reason, row["id"]))
                    log.event("extract", url=row["url"], skipped=reason)
                    continue
                try:
                    found = extract_document(
                        llm, target, clean_text(cached["body"]), source_type=row["source_type"],
                        doc_date=row["published_at"], url=row["url"],
                        max_chunks=cfg.limits.max_chunks_per_page,
                        cache=FileCache(root / "cache" / "llm"), log=log,
                        take_call=lambda: budget.take("llm_calls", "extract"))
                except OverBudget:
                    stopped.append("budget")
                    return
                except ProviderDown as e:
                    add_reason(reasons, "partial_failure", "extract",
                               f"extraction stopped: {e}. Documents already done are kept")
                    stopped.append("provider")
                    return
                except Malformed as e:
                    with conn:
                        conn.execute("UPDATE sources SET stage = 'failed', last_error = ? WHERE id = ?",
                                     ("extraction: the model's output was invalid twice", row["id"]))
                    counters["documents_skipped"] = counters.get("documents_skipped", 0) + 1
                    add_reason(reasons, "partial_failure", "extract",
                               "the model's output was invalid twice; document skipped", url=row["url"])
                    log.event("extract", url=row["url"], skipped=f"malformed output: {str(e)[:200]}")
                    save_run()
                    continue
                # A third-party profile, directory or summary page is not original
                # reporting: it drops to tier 4 and can never make a role current alone.
                lowered = row["tier"] == 3 and found.page_kind == "profile_or_directory"
                basis = "published" if row["published_at"] else "retrieved"
                stored = store_document(
                    conn, company_id, row["id"], found.accepted,
                    version=(f"{min(found.versions)}+{code_version()}" if found.versions
                             else min(versions)), page_kind=found.page_kind,
                    as_of_date=(row["published_at"] or row["retrieved_at"])[:10], as_of_basis=basis,
                    tier=4 if lowered else None)
                counters["documents_extracted"] = counters.get("documents_extracted", 0) + 1
                counters["claims_rejected"] = counters.get("claims_rejected", 0) + len(found.rejected)
                log.event("extract", url=row["url"], claims=stored, rejected=len(found.rejected),
                          page_kind=found.page_kind, about_target=found.about_target,
                          tier_lowered=lowered, llm_calls=found.calls)
                save_run()

        def resolve_now() -> tuple[bool, int]:
            """Rebuild roles from all claims. Returns (a current CEO exists, in-scope people)."""
            drop_people_without_claims(conn, company_id)
            rebuild_roles(conn, company_id, today, cfg.limits.freshness_window_months)
            marks = ",".join("?" * len(IN_SCOPE))
            ceo = conn.execute("SELECT count(*) FROM roles WHERE company_id = ? AND status = 'current' "
                               "AND canonical_role = 'CEO'", (company_id,)).fetchone()[0] > 0
            def leaders(status: str) -> int:
                return conn.execute(
                    f"SELECT count(DISTINCT person_id) FROM roles WHERE company_id = ? AND status = ? "
                    f"AND level IN ({marks})", (company_id, status, *IN_SCOPE)).fetchone()[0]

            # Kept apart everywhere: an unverified role is a lead, never a confirmed leader.
            counters["current_leaders"] = leaders("current")
            counters["unverified_leaders"] = conn.execute(
                f"SELECT count(DISTINCT person_id) FROM roles WHERE company_id = ? AND status = 'unverified' "
                f"AND level IN ({marks}) AND person_id NOT IN (SELECT person_id FROM roles WHERE "
                f"company_id = ? AND status = 'current' AND level IN ({marks}))",
                (company_id, *IN_SCOPE, company_id, *IN_SCOPE)).fetchone()[0]
            return ceo, counters["current_leaders"]

        if extract and llm is not None:
            extract_pending()
            ceo, people = resolve_now()
            # Round 2 runs only when round 1 found too little (SPEC 3.10).
            trigger = None if stopped else ("no current CEO found" if not ceo else
                                            f"only {people} current leaders found" if people < 3 else None)
            if not trigger:
                why = "extraction stopped early" if stopped else (
                    f"a current CEO and {people} current leaders found; "
                    f"{counters['unverified_leaders']} more people are unverified and not counted")
                notes.append(f"search round 2: not needed ({why})")
                log.event("search_round", round=2, ran=False, reason=why)
            elif not search.available or cfg.limits.max_search_rounds < 2:
                notes.append(f"search round 2: wanted ({trigger}) but search is off")
                log.event("search_round", round=2, ran=False, reason="search is not available", trigger=trigger)
            elif counters.get("search_rounds", 0) >= 2 and not refresh:
                notes.append(f"search round 2: already done for this run ({trigger})")
                log.event("search_round", round=2, ran=False, reason="already done", trigger=trigger)
            else:
                before = conn.execute("SELECT count(*) FROM claims WHERE company_id = ?",
                                      (company_id,)).fetchone()[0]
                extra = search_for(round_two_queries(site.name, home_domain), "search round 2")
                counters["search_rounds"] = 2
                plan.extend(extra)
                fetch(extra)
                extract_pending()
                ceo, people = resolve_now()
                after = conn.execute("SELECT count(*) FROM claims WHERE company_id = ?",
                                     (company_id,)).fetchone()[0]
                notes.append(f"search round 2 ran because {trigger}: {len(extra)} pages, "
                             f"{after - before} more claims. No further round is ever run")
                log.event("search_round", round=2, ran=True, trigger=trigger, pages=len(extra),
                          new_claims=after - before)
            # ---------------------------------------------- verify and enrich
            if verify and not stopped:
                def targeted_search(queries: list[str], label: str) -> list[Planned]:
                    if not search.available or cfg.limits.max_search_rounds < 1:
                        return []
                    found = search_for(queries, label, limit=3)     # a check reads at most 3 new pages
                    plan.extend(found)
                    return found

                ops = Ops(conn=conn, company_id=company_id, company=site.name, domain=home_domain, llm=llm,
                          cache=FileCache(root / "cache" / "llm"), pages=fetcher.cache, log=log,
                          counters=counters, reasons=reasons, notes=notes, limits=cfg.limits,
                          take=budget.take, search=targeted_search, fetch=fetch, extract=extract_pending,
                          resolve=resolve_now, add_reason=add_reason, stopped=stopped)
                passed = conn.execute("SELECT verified_pass_at FROM companies WHERE id = ?",
                                      (company_id,)).fetchone()[0]
                # One automatic search pass per company. --refresh allows another; a role
                # already checked is never searched again either way.
                ops.allow_search = refresh or passed is None
                if not ops.allow_search:
                    notes.append(f"targeted searches: none, one full pass was made on {passed[:10]} "
                                 "(run with --refresh to search again)")
                # Plan step 1: role words, searched on the company's own domains only. Once
                # per company, like every automatic search; --refresh allows it again.
                role_done = conn.execute("SELECT role_search_at FROM companies WHERE id = ?",
                                         (company_id,)).fetchone()[0]
                if search.available and cfg.limits.max_search_rounds >= 1 and (refresh or role_done is None):
                    results, complete = [], True
                    for q in role_word_queries(site.name):
                        cached = search.cache.get(f"tavily|{search.max_results}|{q}|domains="
                                                  + ",".join(sorted(site.domains))) and not refresh
                        if not cached and not budget.take("search_queries", "discover"):
                            complete = False
                            break
                        try:
                            found_now, _ = search.query(q, include_domains=sorted(site.domains))
                        except SearchError as e:
                            add_reason(reasons, "partial_failure", "discover", f"role-word search failed: {e}",
                                       query=q)
                            complete = False
                            break
                        results += found_now
                    known = {r["url"] for r in conn.execute("SELECT url FROM sources WHERE company_id = ?",
                                                            (company_id,))}
                    new: list[Planned] = []
                    for r in results:
                        host = registrable(urlsplit(r["url"]).netloc)
                        # The provider's filter is not trusted: anything off the company's domains is dropped.
                        if host in site.domains and r["url"] not in known and host not in _NO_FETCH \
                                and r["url"] not in {p.url for p in new}:
                            new.append(Planned(r["url"], *classify(r["url"], site.domains)))
                    new = new[:ROLE_SEARCH_PAGES]
                    plan.extend(new)
                    fetch(new)
                    extract_pending()
                    resolve_now()
                    if complete:
                        with conn:
                            conn.execute("UPDATE companies SET role_search_at = ? WHERE id = ?", (_now(), company_id))
                    notes.append(f"role-word search on the company's own domains: {len(results)} results, "
                                 f"{len(new)} new pages read")
                    log.event("role_search", results=len(results), new_pages=len(new), complete=complete)
                recover_earlier_titles(ops)     # "had been our SVP": no longer held, by the company's word
                resolve_now()
                recover_employment_end(ops)     # "remain employed until <date>": ends every role then
                recover_dates(ops)          # a date a held source states, without model or search
                resolve_now()
                verify_roles(ops)           # each uncertain role: search, extract, and the rules decide
                recover_dates(ops)
                enrich(ops)                 # location and biography of current leaders
                extract_pending()           # pages fetched while enriching
                ceo, people = resolve_now()
                capped = any(r["code"] == "budget_exhausted" for r in reasons)
                if ops.allow_search and not stopped and not capped:
                    with conn:
                        conn.execute("UPDATE companies SET verified_pass_at = ? WHERE id = ?", (_now(), company_id))
                    notes.append("targeted searches: full pass complete; later runs make no automatic search")
            for problem in problems[reported:]:     # model trouble during round 2 or later
                add_reason(reasons, "partial_failure", "discover", problem)
                notes.append("degraded: " + problem)
            counters["sources_found"] = len(plan)
            if not stopped:
                if not ceo:
                    add_reason(reasons, "insufficient_data", "extract", "no current CEO was found")
                if people == 0:
                    add_reason(reasons, "insufficient_data", "extract",
                               "no current leader (C-level, VP or Head) was found")
            counters["people"] = conn.execute("SELECT count(*) FROM people WHERE company_id = ?",
                                              (company_id,)).fetchone()[0]
            counters["claims"] = conn.execute("SELECT count(*) FROM claims WHERE company_id = ?",
                                              (company_id,)).fetchone()[0]
        elif extract:
            notes.append("extraction: skipped, no LLM key is configured")
            add_reason(reasons, "insufficient_data", "extract",
                       "no LLM key is configured, so no people could be extracted")

        # ----------------------------------------------------------------- end
        have = conn.execute(
            "SELECT count(*) FROM sources WHERE company_id = ? AND stage IN ('fetched','extracted') "
            "AND source_type IN ('company_site','press_release')", (company_id,)).fetchone()[0]
        if have <= 1:       # the homepage alone says little about leadership
            add_reason(reasons, "insufficient_data", "discover",
                       "no company page other than the homepage could be read")
        counters["network_calls_this_process"] = fetcher.network_calls + search.network_calls
        status = headline(reasons)
        save_run(status)
        conn.execute("UPDATE companies SET collected_at = ? WHERE id = ?", (_now(), company_id))
        conn.commit()
        log.event("run_end", run_id=run_id, status=status, reasons=reasons, counters=counters,
                  limits={k: getattr(cfg.limits, v) for k, v in Budget.LIMITS.items()})
        return RunResult(run_id, site.name, status, reasons, counters,
                         fetcher.network_calls + search.network_calls, log.path, notes)
    finally:
        fetcher.close()
        search.close()
        conn.close()


def _db_path(root: Path) -> Path:
    path = root / "data" / "leadership.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def print_report(result: RunResult, cfg: Config, conn_path: Path) -> None:
    """Usage against each limit, every reason, and what was stored."""
    c = result.counters
    print(f"\nCompany:  {result.company}")
    print(f"Run:      {result.run_id}    Status: {result.status}")
    for note in result.notes:
        print(f"  - {note}")
    print("\nUsage against limits")
    for name, limit in Budget.LIMITS.items():
        print(f"  {name:<16}{c.get(name, 0):>4} of {getattr(cfg.limits, limit)}")
    print(f"  {'pages failed':<16}{c.get('pages_failed', 0):>4}")
    print(f"  {'pages skipped':<16}{c.get('pages_skipped', 0):>4}   (refused, robots.txt, or not verified as this company)")
    print(f"  {'network calls':<16}{result.network_calls:>4}   (this process; 0 means all from cache)")
    if result.reasons:
        print("\nReasons recorded")
        for r in result.reasons:
            where = f"  [{r['url']}]" if r.get("url") else ""
            print(f"  {r['code']:<18}{r['stage']:<9}{r['detail']}{where}")
    conn = connect(conn_path)
    if "claims" in c:
        print(f"\nExtraction: {c.get('claims', 0)} claims stored about {c.get('people', 0)} people; "
              f"{c.get('claims_rejected', 0)} rejected by the checks this run; "
              f"{c.get('documents_skipped', 0)} documents skipped")
        people = conn.execute(
            """SELECT person, title, level, status, status_reason, conflict, source_type, source_date
                 FROM leadership WHERE company = ? AND level IN ('c_level', 'vp', 'head')
                ORDER BY CASE status WHEN 'current' THEN 0 WHEN 'announced' THEN 1
                         WHEN 'unverified' THEN 2 ELSE 3 END, level, person""",
            (result.company,)).fetchall()
        print(f"Current leaders: {c.get('current_leaders', 0)}. Unverified, not counted as current: "
              f"{c.get('unverified_leaders', 0)}")
        print("\nLeadership found (C-level, VP, Head)")
        for r in people:
            flag = "  CONFLICT" if r["conflict"] else ""
            print(f"  {r['status']:<11}{r['person']:<24}{r['title'][:44]:<46}"
                  f"{r['source_type'] or '':<14}{r['source_date'] or '':<11}{flag}")
        if not people:
            print("  (none)")
        checks = conn.execute(
            """SELECT k.kind, k.outcome, count(*) AS n FROM checks k JOIN companies c ON c.id = k.company_id
                WHERE c.name = ? GROUP BY 1, 2 ORDER BY 1, 2""", (result.company,)).fetchall()
        if checks:
            print("\nTargeted checks on record (all runs)")
            for k in checks:
                print(f"  {k['kind']:<10}{k['outcome']:<20}{k['n']:>3}")
    rows = conn.execute(
        """SELECT s.source_type, s.tier, s.stage, count(*) AS n FROM sources s
             JOIN companies c ON c.id = s.company_id WHERE c.name = ?
            GROUP BY 1, 2, 3 ORDER BY 2, 1, 3""", (result.company,)).fetchall()
    print("\nSources stored")
    for r in rows:
        print(f"  tier {r['tier']}  {r['source_type']:<14}{r['stage']:<10}{r['n']:>3}")
    unread = conn.execute(
        """SELECT s.url, s.tier, s.source_type, s.stage, s.last_error FROM sources s
             JOIN companies c ON c.id = s.company_id
            WHERE c.name = ? AND s.stage IN ('skipped', 'failed') ORDER BY s.tier, s.url""",
        (result.company,)).fetchall()
    if unread:
        # Shown on every run, including one answered from the cache.
        print("\nPages that could not be read (nothing from them is used)")
        for r in unread:
            own = "  <- the company's own page or filing" if r["tier"] <= 2 else ""
            print(f"  tier {r['tier']}  {r['source_type']:<14}{r['url']}{own}")
            print(f"          {r['stage']}: {r['last_error']}")
    conn.close()
    if result.log_path:
        print(f"\nRun log: {result.log_path}")
