"""Agent 2 - Summary Agent: turns the research package into a structured research brief."""

from __future__ import annotations

import logging

from app.agents.base import AgentContext, AgentError
from app.demo.fixtures import demo_brief
from app.llm_client import LLMError, complete_json
from app.models import ResearchBrief, ResearchPackage
from app.prompts import SUMMARY_SYSTEM, summary_prompt

logger = logging.getLogger(__name__)


def strip_unknown_evidence_ids(data: dict, valid_ids: set[str], topic: str) -> dict:
    """Drop source IDs the model invented; drop evidence left with no real source."""
    if not isinstance(data, dict):
        raise ValueError("brief must be a JSON object")
    kept, removed = [], 0
    for item in data.get("supporting_evidence") or []:
        if not isinstance(item, dict):
            continue
        ids = [i for i in item.get("source_ids") or [] if i in valid_ids]
        if ids:
            kept.append({**item, "source_ids": ids})
        else:
            removed += 1
    limitations = list(data.get("limitations") or [])
    if removed:
        limitations.append(f"{removed} evidence item(s) were removed because they cited no retrieved source.")
    # Free models sometimes return terms as plain strings; normalise when possible.
    terms = []
    for t in data.get("important_terms") or []:
        if isinstance(t, dict):
            terms.append(t)
        elif isinstance(t, str) and ":" in t:
            term, definition = t.split(":", 1)
            terms.append({"term": term, "definition": definition})
    return {**data, "topic": topic, "supporting_evidence": kept, "limitations": limitations,
            "important_terms": terms}


class SummaryAgent:
    name = "Summary Agent"

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    def run(self, package: ResearchPackage) -> ResearchBrief:
        if self.ctx.demo:
            return demo_brief(package.topic, package.sources)
        llm = self.ctx.require_llm()
        valid = package.source_ids
        try:
            brief = complete_json(
                llm, SUMMARY_SYSTEM, summary_prompt(package), ResearchBrief, purpose="summary.brief",
                max_tokens=6000, context={"valid_source_ids": valid},
                prepare=lambda d: strip_unknown_evidence_ids(d, valid, package.topic),
            )
        except LLMError as exc:
            raise AgentError(f"Summary Agent failed: {exc}") from None
        if package.research_status != "sufficient":
            brief.limitations.append("The underlying research was limited; conclusions are tentative.")
        logger.info("Brief: %d key points, %d evidence items", len(brief.key_points), len(brief.supporting_evidence))
        return brief
