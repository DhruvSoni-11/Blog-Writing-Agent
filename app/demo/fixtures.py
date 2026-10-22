"""SIMULATED demonstration data.

Everything in this module is fake, topic-templated placeholder content used only by the explicit
``--demo`` mode. Every record is marked ``simulated`` and every title/snippet starts with
"[SIMULATED]". URLs use the reserved ``.invalid`` top-level domain (RFC 2606), so they can never be
mistaken for (or resolve to) real web pages.
"""

from __future__ import annotations

from app.models import ClaimCheck, EvidenceItem, Finding, ResearchBrief, Source, TermDefinition
from app.tools.web_search import RawResult

DEMO_PROVIDER = "demo-fixtures (simulated)"
DEMO_MODEL = "deterministic-demo-templates (no LLM)"
DEMO_BANNER = (
    "DEMONSTRATION MODE: Using simulated data. No live web research or real factual "
    "verification is being performed."
)

_FIXTURE_SOURCES = [
    ("overview", "Overview of {topic}",
     "An introductory overview describing what {topic} is, its core components and why practitioners care about it."),
    ("docs", "Core concepts of {topic} (documentation)",
     "Documentation-style page that defines the key terms of {topic} and explains step by step how the main components interact."),
    ("case-study", "Case study: applying {topic} in practice",
     "Describes an example project that used {topic}, the problem it addressed and the lessons the team learned."),
    ("survey", "Research survey on {topic}",
     "A survey notes that several studies report benefits from {topic}, but cautions that evaluation methods differ between studies."),
    ("critique", "A critical perspective on {topic}",
     "Argues that the benefits of {topic} are sometimes overstated and highlights cost, reliability and human-oversight concerns."),
    ("guide", "Getting started with {topic}",
     "Practical recommendations for adopting {topic}: start with a small pilot, measure outcomes and keep humans reviewing results."),
]


class FixtureSearchProvider:
    """Offline stand-in for a search engine. Returns the same simulated results for any query."""

    name = DEMO_PROVIDER

    def __init__(self, topic: str) -> None:
        self.topic = topic

    def search(self, query: str, max_results: int) -> list[RawResult]:
        return [
            RawResult(
                title=f"[SIMULATED] {title.format(topic=self.topic)}",
                url=f"https://{slug}.demo.blogforge.invalid/simulated-source",
                snippet=f"[SIMULATED DEMO DATA] {snippet.format(topic=self.topic)}",
                published_date=None,  # simulated records never carry invented dates
            )
            for slug, title, snippet in _FIXTURE_SOURCES
        ]


_SLUGS = [slug for slug, _, _ in _FIXTURE_SOURCES]


def _ids(sources: list[Source], *indexes: int) -> list[str]:
    """Source IDs of fixture rows (by position in _FIXTURE_SOURCES), robust to credibility re-ordering."""
    wanted = [f"https://{_SLUGS[i]}.demo.blogforge.invalid/" for i in indexes]
    found = [s.source_id for s in sources if any(s.url.startswith(w) for w in wanted)]
    return found or [sources[0].source_id]


def demo_findings(topic: str, sources: list[Source]) -> list[Finding]:
    rows = [
        ("key_fact", "retrieved_evidence", (0,), f"[Simulated] {topic} is described as a set of components working together toward a goal."),
        ("concept", "retrieved_evidence", (1,), f"[Simulated] Documentation defines the key terms of {topic} and how its parts interact."),
        ("example", "retrieved_evidence", (2,), f"[Simulated] A case study shows {topic} applied to a concrete problem, with lessons learned."),
        ("key_fact", "retrieved_evidence", (3,), f"[Simulated] Several studies reportedly find benefits, but their evaluation methods differ."),
        ("disagreement", "retrieved_evidence", (3, 4), f"[Simulated] Sources disagree on how large the benefits of {topic} are."),
        ("key_fact", "retrieved_evidence", (5,), "[Simulated] Guides recommend small pilots, measuring outcomes and human review."),
    ]
    return [
        Finding(finding_id=f"F{i}", category=cat, basis=basis, statement=text, source_ids=_ids(sources, *idx))
        for i, (cat, basis, idx, text) in enumerate(rows, start=1)
    ]


def demo_brief(topic: str, sources: list[Source]) -> ResearchBrief:
    return ResearchBrief(
        topic=topic,
        executive_summary=(
            f"[Simulated brief] {topic} combines several components to reach a goal. Simulated sources "
            "describe its core concepts, give a practical example and disagree about the size of its benefits."
        ),
        key_points=[
            f"{topic} is made of components that work together.",
            "Key terms are defined in documentation-style sources.",
            "Reported benefits vary and evaluation methods differ.",
            "Start small, measure results and keep humans in the loop.",
        ],
        supporting_evidence=[
            EvidenceItem(point=f"Overview of what {topic} is and why it matters.", source_ids=_ids(sources, 0)),
            EvidenceItem(point="Definitions of key terms and component interactions.", source_ids=_ids(sources, 1)),
            EvidenceItem(point="A worked example with lessons learned.", source_ids=_ids(sources, 2)),
            EvidenceItem(point="Benefits are reported but measured inconsistently.", source_ids=_ids(sources, 3, 4)),
            EvidenceItem(point="Adoption advice: pilot, measure, review.", source_ids=_ids(sources, 5)),
        ],
        important_terms=[
            TermDefinition(term="Component", definition="A distinct part of the system with one clear job."),
            TermDefinition(term="Pilot", definition="A small, low-risk trial used to learn before scaling up."),
        ],
        examples=[f"A team piloting {topic} on one well-defined problem (simulated case study)."],
        disagreements=["Simulated sources disagree on how large the benefits are."],
        limitations=["All data in this brief is simulated for demonstration purposes."],
        recommended_blog_angle=f"A balanced, beginner-friendly explanation of {topic} with practical advice.",
    )


def demo_claim_checks(topic: str, sources: list[Source]) -> list[ClaimCheck]:
    rows = [
        ("supported", (0, 1), f"{topic} is made of components that work together toward a goal.",
         "Simulated overview and documentation sources state this directly.", None),
        ("supported", (2,), f"A case study describes {topic} being applied to a concrete problem.",
         "The simulated case-study source describes this.", None),
        ("partially_supported", (3,), f"Studies show that {topic} delivers benefits.",
         "The simulated survey says benefits are reported, but methods differ.",
         "Say that several studies report benefits, while noting that methods differ."),
        ("contradictory", (3, 4), f"The benefits of {topic} always outweigh its costs.",
         "The simulated critique disputes this; the survey does not claim it.",
         "Omit, or present it as a debated question."),
        ("unsupported", (), f"{topic} improves team productivity by 40 percent on average.",
         "No simulated source contains this figure.", "Omit: no source supports a specific figure."),
        ("unverifiable", (), f"Most organisations will adopt {topic} within five years.",
         "The simulated sources do not address adoption forecasts.", "Omit or label clearly as speculation."),
    ]
    return [
        ClaimCheck(claim_id=f"C{i}", status=status, supporting_source_ids=_ids(sources, *idx) if idx else [],
                   claim_text=f"[Simulated] {text}", explanation=f"[Simulated check] {why}", recommended_correction=fix)
        for i, (status, idx, text, why, fix) in enumerate(rows, start=1)
    ]


DEMO_LIMITATIONS = [
    "DEMONSTRATION MODE: all sources and verdicts are simulated; no real fact-checking was performed.",
]


def demo_blog_markdown(topic: str, brief: ResearchBrief, include_unsupported_claim: bool) -> str:
    """Deterministic template blog body. On the first pass it deliberately includes one claim the
    (simulated) verification report marked unsupported, so the demo exercises the correction loop."""
    cite = {e.point: e.source_ids for e in brief.supporting_evidence}
    ev = list(cite.items())

    def c(i: int) -> str:
        return " ".join(f"[{sid}]" for sid in ev[min(i, len(ev) - 1)][1])

    terms = "\n\n".join(f"### {t.term}\n\n{t.definition}" for t in brief.important_terms)
    risky = (f" Some advocates say {topic} improves team productivity by 40 percent on average."
             if include_unsupported_claim else "")
    return f"""## Introduction

This post is a demonstration generated from simulated data. {brief.executive_summary} The goal here
is to show how BlogForge AI structures a post, cites its sources and follows a verification report,
not to provide real information about {topic}.

## Understanding the Basics

At a high level, {topic} is described as a set of components that work together toward a goal {c(0)}.
Rather than one monolithic piece, each part has a clear job, and the value comes from how the parts
are combined.{risky}

## Key Concepts

Simulated documentation defines the core vocabulary and explains how the main components interact {c(1)}.

{terms}

## {topic} in Practice

### An Example

A simulated case study describes a team applying {topic} to one concrete, well-defined problem and
recording the lessons it learned along the way {c(2)}. Examples like this are useful because they
show trade-offs that overviews tend to skip.

### What the Evidence Says About Benefits

Several studies reportedly find benefits, although the simulated survey cautions that their
evaluation methods differ, which makes results hard to compare {c(3)}. A critical perspective argues
that benefits are sometimes overstated and raises cost, reliability and oversight concerns. Whether
the benefits always outweigh the costs is therefore a debated question rather than a settled fact.

## Practical Implications

If you are considering {topic}, the simulated guidance is consistent: start with a small pilot,
decide in advance how you will measure outcomes, and keep humans reviewing the results {c(4)}.
Treat forecasts about how quickly organisations will adopt it as speculation until real evidence
is available.

## Conclusion

{brief.recommended_blog_angle} In this demonstration every source and verification verdict is
simulated, so none of the statements above should be treated as researched facts. Run BlogForge AI
in live mode with OpenRouter and a search provider configured to produce a real, referenced post.
"""
