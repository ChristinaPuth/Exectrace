"""Polite, cached page fetching. SPEC section 3.2.

Every page goes through `Fetcher.get`:

- the page cache is read first, so a repeated run makes no network call;
- robots.txt is obeyed, and requests to one host are spaced out;
- timeouts, 429 and 5xx are retried with backoff, honoring Retry-After;
- the raw body is stored, so a change to text cleaning never needs a re-fetch.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from ..cache import FileCache
from ..config import Limits
from ..runlog import RunLog

USER_AGENT = "ExecTraceBot/0.1 (company leadership research; obeys robots.txt)"
_MAX_RETRY_AFTER = 60.0
_MAX_BODY_CHARS = 5_000_000     # a filing can be large; anything beyond this is cut


@dataclass
class Page:
    url: str
    status: int                 # HTTP status; 0 when no response was received
    body: str = ""
    final_url: str = ""
    content_type: str = ""
    retrieved_at: str = ""      # ISO time of the real request, also for cache hits
    from_cache: bool = False
    error: str | None = None    # set when the page could not be used

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None


class Fetcher:
    def __init__(self, limits: Limits, cache_dir: Path, log: RunLog | None = None, *,
                 refresh: bool = False, transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.limits = limits
        self.cache = FileCache(Path(cache_dir) / "pages")
        self.log = log or RunLog(None)
        self.refresh = refresh
        self.network_calls = 0
        self._sleep, self._clock = sleep, clock
        self._client = httpx.Client(transport=transport, follow_redirects=True,
                                    timeout=limits.request_timeout_seconds,
                                    headers={"User-Agent": USER_AGENT})
        self._last_request: dict[str, float] = {}
        self._robots: dict[str, RobotFileParser | None] = {}
        self._refreshed: set[str] = set()
        self._unreachable: set[str] = set()     # hosts whose robots.txt got no usable answer

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------ robots
    def allowed(self, url: str, user_agent: str = USER_AGENT) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            page = self.get(origin + "/robots.txt", check_robots=False)
            if page.status == 200:
                rp = RobotFileParser()
                rp.parse(page.body.splitlines())
                self._robots[origin] = rp
            elif page.status in (401, 403):
                self._robots[origin] = False        # the site refuses this client altogether
            elif 400 <= page.status < 500:
                self._robots[origin] = None         # no robots file: everything is allowed
            else:
                self._robots[origin] = False        # unreachable: assume nothing is allowed
                self._unreachable.add(origin)
        rp = self._robots[origin]
        if rp is None:
            return True
        return bool(rp) and rp.can_fetch(user_agent, url)

    # ------------------------------------------------------------------- fetch
    def get(self, url: str, *, headers: dict[str, str] | None = None,
            check_robots: bool = True) -> Page:
        use_cache = not self.refresh or url in self._refreshed
        if use_cache and (hit := self.cache.get(url)):
            self.log.event("fetch", url=url, status=hit["status"], cache="hit")
            return Page(**hit, from_cache=True)
        agent = (headers or {}).get("User-Agent", USER_AGENT)
        if check_robots and not self.allowed(url, agent):
            parts = urlsplit(url)
            down = f"{parts.scheme}://{parts.netloc}" in self._unreachable
            error = ("the site could not be reached (robots.txt gave no answer)" if down
                     else "disallowed by robots.txt")
            self.log.event("fetch", url=url, status=0, cache="miss", error=error)
            return Page(url=url, status=0, error=error)

        host = urlsplit(url).netloc
        started = self._clock()
        attempts, status, error, response = 0, 0, None, None
        while True:
            wait = self._last_request.get(host, -1e9) + self.limits.host_delay_seconds - self._clock()
            if wait > 0:
                self._sleep(wait)
            attempts += 1
            self.network_calls += 1
            self._last_request[host] = self._clock()
            try:
                response = self._client.get(url, headers=headers)
                status, error = response.status_code, None
            except httpx.HTTPError as e:
                response, status, error = None, 0, type(e).__name__
            retryable = response is None or status == 429 or status >= 500
            if not retryable or attempts > self.limits.max_retries:
                break
            self._sleep(_retry_delay(response, attempts))

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if response is None:
            page = Page(url=url, status=0, retrieved_at=now, error=f"no response: {error}")
        else:
            page = Page(url=url, status=status, body=response.text[:_MAX_BODY_CHARS],
                        final_url=str(response.url), retrieved_at=now,
                        content_type=response.headers.get("content-type", ""),
                        error=None if status == 200 else f"HTTP {status}")
        # A definite answer from the server is cached, including 403 and 404, so
        # it is not requested again. Timeouts, 429 and 5xx are not cached.
        if response is not None and status != 429 and status < 500:
            entry = {k: v for k, v in page.__dict__.items() if k != "from_cache"}
            self.cache.put(url, entry)
            self._refreshed.add(url)
        self.log.event("fetch", url=url, status=status, cache="miss", retries=attempts - 1,
                       latency_s=round(self._clock() - started, 2), error=page.error)
        return page


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    """Retry-After if the server sent one, else exponential backoff: 1, 2, 4 s."""
    if response is not None:
        try:
            return min(float(response.headers.get("retry-after", "")), _MAX_RETRY_AFTER)
        except ValueError:
            pass
    return float(2 ** (attempt - 1))


# ------------------------------------------------------------------ page text
_DATE = r"(\d{4}-\d{2}-\d{2})"
_PUBLISHED = [
    re.compile(r'<meta[^>]+(?:property|name)=["\'](?:article:published_time|og:published_time|'
               r'datePublished|date|pubdate|publish-date|parsely-pub-date)["\'][^>]*'
               r'content=["\']' + _DATE, re.I),
    re.compile(r'<meta[^>]+content=["\']' + _DATE + r'[^>]*(?:property|name)=["\']'
               r'(?:article:published_time|og:published_time|datePublished)["\']', re.I),
    re.compile(r'"datePublished"\s*:\s*"' + _DATE),
    re.compile(r'<time[^>]+datetime=["\']' + _DATE, re.I),
]


def find_published(html: str) -> str | None:
    """The publication date a page states about itself, or None. Never guessed."""
    for pattern in _PUBLISHED:
        if m := pattern.search(html):
            return m.group(1)
    return None


def find_title(html: str) -> str | None:
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    return re.sub(r"\s+", " ", html_lib.unescape(m.group(1))).strip()[:300] if m else None


def clean_text(html: str) -> str:
    """Readable text of a page. Falls back to tag stripping for card-style pages."""
    text = ""
    try:
        import trafilatura
        text = trafilatura.extract(html, include_comments=False, include_tables=True,
                                   favor_recall=True) or ""
    except Exception:       # a malformed page must never stop a run
        text = ""
    if len(text) < 200:
        stripped = re.sub(r"(?is)<(script|style|noscript|svg|template)\b.*?</\1>", " ", html)
        stripped = html_lib.unescape(re.sub(r"(?s)<[^>]+>", " ", stripped))
        stripped = re.sub(r"\s+", " ", stripped).strip()
        text = max(text, stripped, key=len)
    return text


def content_hash(html: str) -> str:
    return hashlib.sha256(clean_text(html).encode("utf-8")).hexdigest()
