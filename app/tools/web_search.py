"""Research service: Tavily (primary) with DuckDuckGo fallback.

Only metadata returned by the search provider is kept (title, URL, snippet, date). Pages are not
scraped. Nothing is ever invented: missing dates stay ``None`` and empty searches raise errors.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from app.config import Settings
from app.models import Source, utc_now_iso
from app.utils.text_utils import domain_of, is_valid_url, normalize_url, sanitize_untrusted

logger = logging.getLogger(__name__)


class SearchError(Exception):
    """Raised when no search provider could return usable results."""


@dataclass
class RawResult:
    title: str
    url: str
    snippet: str = ""
    published_date: str | None = None


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int) -> list[RawResult]: ...


# --------------------------------------------------------------------------- providers
class TavilyProvider:
    name = "tavily"

    def __init__(self, api_key: str, timeout: float = 30) -> None:
        from tavily import TavilyClient

        self._client = TavilyClient(api_key=api_key)
        self._timeout = timeout

    def search(self, query: str, max_results: int) -> list[RawResult]:
        data = self._client.search(query, search_depth="basic", max_results=max_results,
                                   include_answer=False, include_raw_content=False, timeout=self._timeout)
        return [
            RawResult(title=r.get("title") or "", url=r.get("url") or "", snippet=r.get("content") or "",
                      published_date=r.get("published_date") or None)
            for r in (data or {}).get("results", [])
        ]


class DuckDuckGoProvider:
    name = "duckduckgo"

    def __init__(self, timeout: float = 30) -> None:
        self._timeout = int(timeout)

    def search(self, query: str, max_results: int) -> list[RawResult]:
        from ddgs import DDGS

        rows = DDGS(timeout=self._timeout).text(query, max_results=max_results, safesearch="moderate")
        return [
            RawResult(title=r.get("title") or "", url=r.get("href") or r.get("url") or "",
                      snippet=r.get("body") or "", published_date=r.get("date") or None)
            for r in rows or []
        ]


# --------------------------------------------------------------------------- credibility
_HIGH = ("arxiv.org", "nature.com", "science.org", "sciencedirect.com", "springer.com", "ieee.org",
         "acm.org", "ncbi.nlm.nih.gov", "who.int", "un.org", "europa.eu", "oecd.org", "nist.gov")
_REFERENCE = ("wikipedia.org", "britannica.com")
_NEWS = ("reuters.com", "apnews.com", "bbc.co.uk", "bbc.com", "nytimes.com", "theguardian.com",
         "ft.com", "economist.com", "wired.com", "technologyreview.com", "arstechnica.com")
_LOW = ("medium.com", "quora.com", "reddit.com", "pinterest.com", "facebook.com", "tiktok.com",
        "x.com", "twitter.com", "linkedin.com")


def classify_source(url: str) -> tuple[str, int]:
    """Return (source_type, rank) where a higher rank means more credible by heuristic."""
    host = domain_of(url)
    parts = host.split(".")

    def matches(domains: tuple[str, ...]) -> bool:
        return any(host == d or host.endswith("." + d) for d in domains)

    if parts[-1] in {"gov", "mil"} or (len(parts) > 2 and parts[-2] in {"gov", "mil"}):
        return "government", 3
    if "edu" in parts or (len(parts) > 2 and parts[-2] == "ac") or matches(_HIGH):
        return "academic_or_official", 3
    if parts[0] in {"docs", "developer", "developers", "learn"} or host.endswith("readthedocs.io"):
        return "documentation", 2
    if matches(_REFERENCE):
        return "reference", 2
    if matches(_NEWS):
        return "reputable_news", 2
    if matches(_LOW):
        return "user_generated", 0
    return "general", 1


# --------------------------------------------------------------------------- service
@dataclass
class SearchOutcome:
    sources: list[Source]
    provider: str
    warnings: list[str] = field(default_factory=list)


_AD_URL_RE = re.compile(
    r"^https?://([a-z0-9.-]*\.)?(bing\.com/aclick|duckduckgo\.com/y\.js|googleadservices\.com|"
    r"doubleclick\.net|google\.com/aclk)", re.IGNORECASE)
_SNIPPET_DATE_RE = re.compile(
    r"^((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4})\s*[·\-–—|]\s*")


def split_snippet_date(snippet: str) -> tuple[str | None, str]:
    """Search engines often prefix snippets with the page date ("Feb 18, 2026 · ..."). Only an explicit
    absolute date is extracted; relative dates such as "1 day ago" are left as text (date stays None)."""
    match = _SNIPPET_DATE_RE.match(snippet or "")
    if not match:
        return None, snippet
    try:
        parsed = datetime.strptime(match.group(1).replace("Sept", "Sep"), "%b %d, %Y")
    except ValueError:
        try:
            parsed = datetime.strptime(match.group(1), "%B %d, %Y")
        except ValueError:
            return None, snippet
    return parsed.date().isoformat(), snippet[match.end():]


def build_sources(results: list[RawResult], provider: str, limit: int, simulated: bool = False) -> tuple[list[Source], list[str]]:
    """Validate, deduplicate (by normalized URL), rank by credibility and number the results."""
    warnings: list[str] = []
    seen: set[str] = set()
    candidates: list[tuple[int, int, RawResult, str]] = []
    rejected = ads = 0
    for order, result in enumerate(results):
        url = (result.url or "").strip()
        if not is_valid_url(url):
            rejected += 1
            continue
        if _AD_URL_RE.match(url):
            ads += 1
            continue
        key = normalize_url(url)
        if key in seen:
            continue
        seen.add(key)
        source_type, rank = classify_source(url)
        candidates.append((rank, order, result, source_type))
    if rejected:
        warnings.append(f"Rejected {rejected} search result(s) with malformed URLs.")
    if ads:
        warnings.append(f"Skipped {ads} sponsored/advertising result(s).")

    candidates.sort(key=lambda c: (-c[0], c[1]))  # credible first, then original search order
    retrieved_at = utc_now_iso()
    sources = []
    for index, (_, _, result, source_type) in enumerate(candidates[:limit], start=1):
        snippet_date, snippet = split_snippet_date(sanitize_untrusted(result.snippet, 700))
        sources.append(Source(
            source_id=f"S{index}",
            title=sanitize_untrusted(result.title, 200) or domain_of(result.url),
            url=result.url.strip(),
            snippet=snippet,
            publication_date=(result.published_date or snippet_date or None),
            retrieved_at=retrieved_at,
            provider=provider,
            source_type=source_type,
            simulated=simulated,
        ))
    return sources, warnings


class ResearchService:
    """Runs queries against the first provider that works; never fabricates results."""

    def __init__(self, providers: list[SearchProvider], max_results: int = 8,
                 delay_seconds: float = 1.0, simulated: bool = False) -> None:
        if not providers:
            raise ValueError("at least one search provider is required")
        self.providers = providers
        self.max_results = max_results
        self.delay_seconds = delay_seconds
        self.simulated = simulated

    @classmethod
    def from_settings(cls, settings: Settings) -> ResearchService:
        providers: list[SearchProvider] = []
        if settings.tavily_configured:
            providers.append(TavilyProvider(settings.tavily_api_key.get_secret_value(), settings.request_timeout))
        providers.append(DuckDuckGoProvider(settings.request_timeout))
        return cls(providers, max_results=settings.max_search_results)

    def research(self, queries: list[str]) -> SearchOutcome:
        queries = [q.strip() for q in queries if q and q.strip()]
        if not queries:
            raise SearchError("No research queries were provided.")
        per_query = min(10, math.ceil(self.max_results / len(queries)) + 2)
        warnings: list[str] = []

        for provider in self.providers:
            results: list[RawResult] = []
            failures = 0
            for index, query in enumerate(queries):
                if index and self.delay_seconds:
                    time.sleep(self.delay_seconds)  # be polite to free search services
                for attempt in (1, 2):  # one bounded retry for transient errors / rate limits
                    try:
                        results.extend(provider.search(query, per_query))
                        break
                    except Exception as exc:  # provider libraries raise many different errors
                        logger.warning("%s search failed for query %r (attempt %d): %s",
                                       provider.name, query, attempt, type(exc).__name__)
                        if attempt == 2:
                            failures += 1
                        elif self.delay_seconds:
                            time.sleep(self.delay_seconds * 2)
            sources, build_warnings = build_sources(results, provider.name, self.max_results, self.simulated)
            if sources:
                if failures:
                    warnings.append(f"{provider.name}: {failures} of {len(queries)} queries failed.")
                logger.info("Search provider %s returned %d sources", provider.name, len(sources))
                return SearchOutcome(sources, provider.name, warnings + build_warnings)
            reason = "all queries failed" if failures == len(queries) else "no usable results"
            warnings.append(f"{provider.name}: {reason}.")
            logger.warning("Search provider %s produced no sources (%s)", provider.name, reason)

        raise SearchError("Web research failed: " + " ".join(warnings) +
                          " Check your network connection or TAVILY_API_KEY, or run with --demo.")
