"""Agent 1 - Research Agent: plans queries, runs web search and extracts evidence-based findings."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.agents.base import AgentContext, AgentError
from app.demo.fixtures import demo_findings
from app.llm_client import LLMError, complete_json
from app.models import Finding, ResearchPackage, Source
from app.prompts import (
    RESEARCH_FINDINGS_SYSTEM,
    RESEARCH_QUERY_SYSTEM,
    research_findings_prompt,
    research_query_prompt,
)
from app.tools.web_search import SearchError

logger = logging.getLogger(__name__)


class _QueryPlan(BaseModel):
    queries: list[str] = Field(min_length=1)


class _FindingsResponse(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def deterministic_queries(topic: str, n: int) -> list[str]:
    templates = ["{t}", "{t} explained", "{t} examples and use cases", "{t} challenges and limitations",
                 "{t} latest research"]
    return [tpl.format(t=topic) for tpl in templates][:n]


def clean_queries(queries: list[str], n: int) -> list[str]:
    seen, cleaned = set(), []
    for q in queries:
        q = " ".join(str(q).split())[:150]
        if q and q.lower() not in seen:
            seen.add(q.lower())
            cleaned.append(q)
    return cleaned[:n]


class ResearchAgent:
    name = "Research Agent"

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    def run(self, topic: str, audience: str | None = None, constraints: str | None = None) -> ResearchPackage:
        topic = (topic or "").strip()
        if not topic:
            raise AgentError("The topic must not be empty.")
        n = self.ctx.settings.max_research_queries
        notes: list[str] = []

        queries = self._plan_queries(topic, audience, constraints, n, notes)
        try:
            outcome = self.ctx.research_service.research(queries)
        except SearchError as exc:
            raise AgentError(str(exc)) from None
        notes.extend(outcome.warnings)

        findings = self._extract_findings(topic, outcome.sources, notes)
        enough = len(outcome.sources) >= 4 and len(findings) >= 3
        status = "sufficient" if enough else "limited"
        if not enough:
            notes.append("Limited evidence: few sources or findings were available; treat the brief with caution.")

        package = ResearchPackage(
            topic=topic, audience=audience, research_queries=queries, sources=outcome.sources,
            findings=findings, search_provider=outcome.provider, research_status=status,
            mode=self.ctx.mode, notes=notes,
        )
        logger.info("Research complete: %d sources, %d findings, status=%s", len(package.sources),
                    len(findings), status)
        return package

    # ------------------------------------------------------------------ steps
    def _plan_queries(self, topic, audience, constraints, n, notes) -> list[str]:
        if self.ctx.demo:
            return deterministic_queries(topic, n)
        llm = self.ctx.require_llm()
        try:
            plan = complete_json(llm, RESEARCH_QUERY_SYSTEM, research_query_prompt(topic, audience, constraints, n),
                                 _QueryPlan, purpose="research.queries", max_tokens=2000)
            queries = clean_queries(plan.queries, n)
            if queries:
                return queries
        except LLMError as exc:
            # Queries are only search strings (not evidence), so a deterministic fallback is safe;
            # it is recorded as a visible warning rather than done silently.
            logger.warning("Query planning failed: %s", exc)
            notes.append(f"LLM query planning failed ({exc}); used deterministic search queries instead.")
        return deterministic_queries(topic, n)

    def _extract_findings(self, topic: str, sources: list[Source], notes: list[str]) -> list[Finding]:
        if self.ctx.demo:
            return demo_findings(topic, sources)
        llm = self.ctx.require_llm()
        valid = {s.source_id for s in sources}
        dropped: list[str] = []

        def prepare(data: dict) -> dict:
            findings = []
            for raw in (data.get("findings") or []):
                if not isinstance(raw, dict):
                    continue
                ids = [i for i in raw.get("source_ids") or [] if i in valid]
                if raw.get("basis", "retrieved_evidence") == "retrieved_evidence" and not ids:
                    dropped.append(str(raw.get("statement", ""))[:80])
                    continue
                findings.append({**raw, "source_ids": ids, "finding_id": f"F{len(findings) + 1}"})
            return {"findings": findings, "notes": [str(n) for n in data.get("notes") or []]}

        try:
            response = complete_json(llm, RESEARCH_FINDINGS_SYSTEM, research_findings_prompt(topic, sources),
                                     _FindingsResponse, purpose="research.findings", max_tokens=6000,
                                     prepare=prepare)
        except LLMError as exc:
            raise AgentError(f"Research Agent could not extract findings: {exc}") from None
        if dropped:
            notes.append(f"Dropped {len(dropped)} finding(s) that cited no retrieved source.")
        notes.extend(response.notes[:5])
        return response.findings
