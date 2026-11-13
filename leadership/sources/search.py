"""Web search through Tavily, cached by query. SPEC sections 3.1 and 6.3.

Optional: without TAVILY_API_KEY the pipeline skips search and says so.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from ..cache import FileCache
from ..config import Limits
from ..runlog import RunLog

ENDPOINT = "https://api.tavily.com/search"
PROVIDER = "tavily"


class SearchError(Exception):
    """Search could not be completed. The message never contains the key."""


def round_one_queries(company: str, domain: str) -> list[str]:
    """Fixed queries. The domain anchors them to this company, not a namesake."""
    return [f'"{company}" {domain} leadership team',
            f'"{company}" {domain} CEO CTO CFO founder',
            f'"{company}" {domain} "vice president"',
            f'"{company}" {domain} "head of"']


class Search:
    def __init__(self, limits: Limits, cache_dir: Path, log: RunLog | None = None, *,
                 refresh: bool = False, transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep, max_results: int = 5):
        self.key = os.environ.get("TAVILY_API_KEY", "").strip()
        self.limits, self.refresh, self.max_results = limits, refresh, max_results
        self.cache = FileCache(Path(cache_dir) / "search")
        self.log = log or RunLog(None)
        self.network_calls = 0
        self._sleep = sleep
        self._client = httpx.Client(transport=transport, timeout=limits.request_timeout_seconds)

    @property
    def available(self) -> bool:
        return bool(self.key)

    def close(self) -> None:
        self._client.close()

    def query(self, q: str) -> tuple[list[dict], bool]:
        """Results as [{title, url, content, published_date}], and whether the cache was used."""
        cache_key = f"{PROVIDER}|{self.max_results}|{q}"
        if not self.refresh and (hit := self.cache.get(cache_key)):
            self.log.event("search", query=q, cache="hit", results=len(hit["results"]))
            return hit["results"], True
        started, attempts = time.monotonic(), 0
        while True:
            attempts += 1
            self.network_calls += 1
            try:
                r = self._client.post(
                    ENDPOINT, headers={"Authorization": f"Bearer {self.key}"},
                    json={"query": q, "max_results": self.max_results, "search_depth": "basic",
                          "include_published_date": True})
                status = r.status_code
            except httpx.HTTPError as e:
                r, status = None, 0
                error = type(e).__name__
            if status == 200:
                break
            retryable = r is None or status == 429 or status >= 500
            if not retryable or attempts > self.limits.max_retries:
                detail = {401: "the search key was rejected", 432: "the search plan's limit is used up",
                          433: "the search plan's limit is used up"}.get(status)
                message = detail or (f"HTTP {status}" if r is not None else f"no response: {error}")
                self.log.event("search", query=q, cache="miss", status=status, retries=attempts - 1,
                               error=message)
                raise SearchError(message)
            try:
                delay = min(float(r.headers.get("retry-after", "")), 60.0) if r is not None else 0
            except ValueError:
                delay = 0
            self._sleep(delay or float(2 ** (attempts - 1)))
        results = [{"title": x.get("title"), "url": x.get("url"), "content": x.get("content"),
                    "published_date": x.get("published_date")}
                   for x in r.json().get("results", []) if x.get("url")]
        self.cache.put(cache_key, {"results": results})
        self.log.event("search", query=q, cache="miss", status=200, retries=attempts - 1,
                       results=len(results), latency_s=round(time.monotonic() - started, 2))
        return results, False
