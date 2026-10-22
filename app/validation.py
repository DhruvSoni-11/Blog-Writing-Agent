"""Deterministic (non-LLM) checks on blog drafts and assembly of the final Markdown file.

References are always generated here from the retrieved sources, never by the model, so every
reference in the output corresponds to a real retrieved result (or a labelled simulated one).
"""

from __future__ import annotations

import re
from datetime import date

from app.models import BlogDraft, FinalBlog, ResearchPackage, ValidationResult, VerificationReport
from app.utils.text_utils import (
    content_tokens,
    normalize_url,
    split_sentences,
    token_similarity,
    word_count,
)

CLAIM_SIMILARITY_THRESHOLD = 0.75
_CITATION_RE = re.compile(r"\[(S\d+(?:\s*[,;]\s*S\d+)*)\]")
_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_STAT_RE = re.compile(r"(\d+(?:\.\d+)?)\s?(?:%|percent\b|per cent\b)", re.IGNORECASE)
_REFS_SECTION_RE = re.compile(r"^#{1,3}\s*(references|sources|bibliography|works cited)\b.*?(?=^##\s|\Z)",
                              re.IGNORECASE | re.MULTILINE | re.DOTALL)
_HEDGES = ("no evidence", "not supported", "unsupported", "unverified", "unclear", "disputed", "debated",
           "contested", "no source", "could not be verified", "cannot be verified", "speculat")


def _h2_headings(body: str) -> list[str]:
    return [m.group(1).strip().lower() for m in re.finditer(r"^##\s+(.+)$", body, re.MULTILINE)]


def cited_ids(body: str) -> list[str]:
    ordered: list[str] = []
    for group in _CITATION_RE.findall(body):
        for sid in re.split(r"\s*[,;]\s*", group):
            if sid not in ordered:
                ordered.append(sid)
    return ordered


def _prose_sentences(body: str) -> list[str]:
    prose = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))
    return split_sentences(prose)


def validate_draft(draft: BlogDraft, package: ResearchPackage, report: VerificationReport,
                   requested_words: int) -> ValidationResult:
    body = draft.body_markdown
    issues: list[str] = []
    warnings: list[str] = []
    flagged: list[str] = []

    # 1. Structure
    headings = _h2_headings(body)
    if not any("introduction" in h for h in headings):
        issues.append("Missing an '## Introduction' section.")
    if not any("conclusion" in h for h in headings):
        issues.append("Missing a '## Conclusion' section.")
    if len(headings) < 4:
        issues.append(f"Only {len(headings)} H2 sections; add clearly organised body sections.")
    if not any(re.search(r"practical|implication|recommend|takeaway|best practice|getting started", h) for h in headings):
        warnings.append("No section on practical implications or recommendations.")
    if not re.search(r"^###\s+", body, re.MULTILINE):
        warnings.append("No H3 sub-headings used.")
    if _REFS_SECTION_RE.search(body):
        warnings.append("Writer included its own references section; it was replaced with verified references.")

    # 2. Citations must point at retrieved sources
    ids = cited_ids(body)
    unknown = [i for i in ids if i not in package.source_ids]
    if unknown:
        issues.append(f"Cites source IDs that were never retrieved: {', '.join(unknown)}. Use only existing IDs.")
    if not [i for i in ids if i in package.source_ids]:
        warnings.append("The draft contains no inline source citations.")

    # 3. URLs must come from research
    known_urls = {normalize_url(s.url) for s in package.sources}
    foreign = sorted({u for u in _URL_RE.findall(body) if normalize_url(u.rstrip(".,;")) not in known_urls})
    if foreign:
        issues.append(f"Contains {len(foreign)} URL(s) that were not retrieved during research; remove them.")

    # 4. Claims the verification report rejected must not be restated as fact
    sentences = _prose_sentences(body)
    for claim in report.material_issues:
        claim_text = re.sub(r"^\[simulated\]\s*", "", claim.claim_text, flags=re.IGNORECASE)
        if len(content_tokens(claim_text)) < 3:
            continue
        for sentence in sentences:
            if any(h in sentence.lower() for h in _HEDGES):
                continue
            if token_similarity(sentence, claim_text) >= CLAIM_SIMILARITY_THRESHOLD:
                issues.append(f"Restates {claim.status} claim {claim.claim_id} ('{claim_text[:90]}'). "
                              f"Correction: {claim.recommended_correction or 'omit it'}.")
                flagged.append(sentence)
                break

    # 5. Statistics must appear in the retrieved evidence
    evidence_text = " ".join(f"{s.title} {s.snippet}" for s in package.sources)
    evidence_stats = {m for m in _STAT_RE.findall(evidence_text)}
    for sentence in sentences:
        for figure in _STAT_RE.findall(sentence):
            if figure not in evidence_stats:
                issues.append(f"Statistic '{figure}%' does not appear in any retrieved source; remove or source it.")
                if sentence not in flagged:
                    flagged.append(sentence)

    # 6. Length (demonstration templates have a fixed length, so they are not checked)
    words = word_count(body)
    if package.mode == "live" and words < requested_words * 0.6 or words > requested_words * 1.6:
        warnings.append(f"Length is {words} words versus about {requested_words} requested.")

    return ValidationResult(material_issues=list(dict.fromkeys(issues)), warnings=warnings,
                            cited_source_ids=[i for i in ids if i in package.source_ids],
                            flagged_sentences=flagged, word_count=words)


# --------------------------------------------------------------------------- final assembly
def _clean_body(body: str, package: ResearchPackage, remove_sentences: list[str]) -> tuple[str, int]:
    removed = 0
    for sentence in remove_sentences:
        if sentence in body:
            body = body.replace(sentence, "", 1)
            removed += 1
    body = _REFS_SECTION_RE.sub("", body)
    known_urls = {normalize_url(s.url) for s in package.sources}

    def fix_link(m: re.Match) -> str:
        return m.group(0) if normalize_url(m.group(2)) in known_urls else m.group(1)

    body = _LINK_RE.sub(fix_link, body)
    body = _URL_RE.sub(lambda m: m.group(0) if normalize_url(m.group(0).rstrip(".,;")) in known_urls else "", body)

    def fix_citation(m: re.Match) -> str:
        valid = [s for s in re.split(r"\s*[,;]\s*", m.group(1)) if s in package.source_ids]
        return "".join(f"[{s}]" for s in valid)

    body = _CITATION_RE.sub(fix_citation, body)
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r" +([.,;:])", r"\1", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip(), removed


def assemble_final_blog(draft: BlogDraft, validation: ValidationResult, package: ResearchPackage,
                        report: VerificationReport, model_name: str, demo_banner: str | None = None) -> FinalBlog:
    warnings = list(validation.warnings)
    to_remove = validation.flagged_sentences if validation.material_issues else []
    body, removed = _clean_body(draft.body_markdown, package, to_remove)
    if removed:
        warnings.append(f"Removed {removed} sentence(s) that still contained unsupported claims after the "
                        "correction allowance was used.")
    for issue in validation.material_issues:
        warnings.append(f"Unresolved after correction pass: {issue}")

    ids = cited_ids(body)
    cited = [package.get_source(i) for i in ids if package.get_source(i)]
    lines = []
    for s in cited or package.sources:
        meta = [s.source_type.replace("_", " ")]
        if s.publication_date:
            meta.append(f"published {s.publication_date}")
        meta.append(f"retrieved {s.retrieved_at[:10]}")
        if s.simulated:
            meta.append("SIMULATED")
        title = s.title.replace("[", "(").replace("]", ")")
        lines.append(f"- **[{s.source_id}]** [{title}]({s.url}) — {', '.join(meta)}")
    ref_heading = "## References" if cited else "## Sources Consulted"
    ref_note = "" if cited else "\n*The post contains no inline citations; these are all sources retrieved during research.*\n"

    header = [f"# {draft.title}", "", f"*{draft.description}*", ""]
    if draft.keywords:
        header += [f"**Keywords:** {', '.join(draft.keywords)}", ""]
    if demo_banner:
        header += [f"> **{demo_banner}**", ""]
    footer = (
        f"\n---\n\n*Generated by BlogForge AI on {date.today().isoformat()} "
        f"(model: {model_name}; search: {package.search_provider}). Verification status: "
        f"{report.overall_status} ({report.claims_checked} claims checked). Automated verification checks "
        "claims against search snippets and does not guarantee factual correctness.*\n"
    )
    markdown = "\n".join(header) + "\n" + body + f"\n\n{ref_heading}\n{ref_note}\n" + "\n".join(lines) + "\n" + footer
    return FinalBlog(title=draft.title, description=draft.description, keywords=draft.keywords,
                     markdown=markdown, word_count=word_count(body),
                     cited_source_ids=[s.source_id for s in cited], warnings=warnings)
