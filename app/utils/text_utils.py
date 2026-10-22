"""Text helpers: URLs, slugs, JSON extraction, word counts and sanitising untrusted text."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING_PARAMS = {"gclid", "fbclid", "mc_cid", "mc_eid", "ref", "ref_src", "igshid", "msclkid"}
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for", "with", "by", "is",
    "are", "was", "were", "be", "been", "it", "its", "this", "that", "these", "those", "as",
    "at", "from", "can", "may", "might", "will", "would", "should", "could", "has", "have",
    "had", "not", "than", "then", "so", "such", "into", "about", "also", "which", "who",
}


# --------------------------------------------------------------------------- URLs
def is_valid_url(url: str | None) -> bool:
    """Return True for absolute http(s) URLs with a plausible host name."""
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    if any(ch.isspace() for ch in url) or len(url) > 2048:
        return False
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if parts.scheme not in {"http", "https"}:
        return False
    host = (parts.hostname or "").lower()
    if not host or "." not in host or host.startswith(".") or host.endswith("."):
        return False
    return re.fullmatch(r"[a-z0-9.\-]+", host) is not None


def normalize_url(url: str) -> str:
    """Canonical form used for de-duplication (lower-case host, no www/fragment/tracking)."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if parts.port and parts.port not in (80, 443):
        host = f"{host}:{parts.port}"
    query = urlencode(
        sorted(
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS
        )
    )
    path = parts.path.rstrip("/") or ""
    # http and https versions of the same page are treated as duplicates.
    return urlunsplit(("https", host, path, query, ""))


def domain_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


# --------------------------------------------------------------------------- Filenames
def slugify(text: str, max_length: int = 50) -> str:
    """Make a filesystem-safe, lower-case slug. Never returns an empty string."""
    normalized = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    slug = slug[:max_length].strip("-")
    # Avoid reserved Windows device names such as CON or NUL.
    if slug.upper() in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(com|lpt)\d", slug):
        slug = f"topic-{slug}"
    return slug or "untitled"


# --------------------------------------------------------------------------- Words
def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def content_tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text or "") if w.lower() not in _STOPWORDS}


def token_similarity(a: str, b: str) -> float:
    """Overlap coefficient of content words (0..1); cheap fuzzy match for claim detection."""
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def extract_numbers(text: str) -> set[str]:
    """Numbers that look like factual figures (percentages, decimals, numbers >= 10)."""
    found = set()
    for match in re.finditer(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(\s?%)?", text or ""):
        raw = match.group(1).replace(",", "")
        if match.group(2) or "." in raw or float(raw) >= 10:
            found.add(raw)
    return found


# --------------------------------------------------------------------------- JSON
def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in an LLM reply (tolerates code fences and prose).

    Raises ValueError if no valid JSON can be found. The result must still be validated.
    """
    if not text or not text.strip():
        raise ValueError("empty response")
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Only try the first "{" and the first "[": scanning further could accept a nested fragment of
    # a truncated reply (e.g. one claim object) as if it were the whole answer.
    decoder = json.JSONDecoder()
    starts = sorted(i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0)
    for index in starts:
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("no complete JSON object found in response (it may have been cut off)")


# --------------------------------------------------------------------------- Untrusted text
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TAG_RE = re.compile(r"</?\s*untrusted_source[^>]*>", re.IGNORECASE)


def sanitize_untrusted(text: str | None, max_chars: int = 700) -> str:
    """Clean retrieved web text before it is shown to a model.

    Removes control characters and anything resembling our own delimiter tags (so a web page
    cannot "close" its data block and smuggle in instructions), collapses whitespace and truncates.
    The text is still untrusted; prompts tell agents to treat it purely as evidence.
    """
    if not text:
        return ""
    text = _CONTROL_RE.sub(" ", str(text))
    text = _TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text
