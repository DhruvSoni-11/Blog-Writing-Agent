"""Agent 3 - Verification Agent: checks the brief's factual claims against retrieved sources.

The LLM proposes a verdict per claim; deterministic rules then make the verdicts more conservative:
unknown source IDs are removed, "supported" claims without sources are downgraded, and claims
whose figures do not appear in any cited source are marked unsupported.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.agents.base import AgentContext, AgentError
from app.demo.fixtures import DEMO_LIMITATIONS, demo_claim_checks
from app.llm_client import LLMError, complete_json
from app.models import ClaimCheck, ResearchBrief, ResearchPackage, VerificationReport
from app.prompts import VERIFICATION_SYSTEM, verification_prompt
from app.utils.text_utils import extract_numbers

logger = logging.getLogger(__name__)

MAX_CLAIMS = 15
STANDARD_LIMITATIONS = [
    "Evidence consists of search-result snippets and extracts, not full articles.",
    "Verdicts are produced by a language model plus simple rules; automated verification does "
    "not guarantee factual correctness.",
]
_STATUSES = {"supported", "partially_supported", "unsupported", "contradictory", "unverifiable"}


class _ClaimsResponse(BaseModel):
    claims: list[ClaimCheck] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


def apply_deterministic_rules(raw: dict, package: ResearchPackage) -> dict:
    """Normalise raw LLM claims and apply conservative, rule-based downgrades."""
    if not isinstance(raw, dict):
        raise ValueError("verification output must be a JSON object")
    valid = package.source_ids
    claims = []
    for item in (raw.get("claims") or [])[:MAX_CLAIMS]:
        if not isinstance(item, dict) or not str(item.get("claim_text", "")).strip():
            continue
        status = str(item.get("status", "unverifiable")).strip().lower().replace(" ", "_")
        if status not in _STATUSES:
            status = "unverifiable"
        ids = [i for i in item.get("supporting_source_ids") or [] if i in valid]
        explanation = str(item.get("explanation") or "No explanation given.").strip()
        correction = item.get("recommended_correction") or None

        if status in ("supported", "partially_supported") and not ids:
            status = "unverifiable"
            explanation += " [Rule: downgraded because no retrieved source was cited.]"
        if status in ("supported", "partially_supported"):
            evidence = " ".join(f"{s.title} {s.snippet}" for s in package.sources if s.source_id in ids)
            missing = sorted(extract_numbers(item["claim_text"]) - extract_numbers(evidence))
            if missing:
                status = "unsupported"
                explanation += f" [Rule: figure(s) {', '.join(missing)} not found in the cited sources.]"
                correction = correction or "Remove the specific figure or state it without a number."
        if status != "supported" and not correction:
            correction = "Use cautious wording or omit this claim."

        claims.append({
            "claim_id": f"C{len(claims) + 1}", "claim_text": item["claim_text"], "status": status,
            "supporting_source_ids": ids, "explanation": explanation, "recommended_correction": correction,
        })
    return {"claims": claims, "limitations": [str(x) for x in raw.get("limitations") or []][:5]}


class VerificationAgent:
    name = "Verification Agent"

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    def run(self, package: ResearchPackage, brief: ResearchBrief) -> VerificationReport:
        if self.ctx.demo:
            checks = demo_claim_checks(package.topic, package.sources)
            return VerificationReport.from_checks(checks, len(package.sources), DEMO_LIMITATIONS, mode="demonstration")

        llm = self.ctx.require_llm()
        try:
            response = complete_json(
                llm, VERIFICATION_SYSTEM, verification_prompt(package, brief), _ClaimsResponse,
                purpose="verification.claims", max_tokens=6000,
                context={"valid_source_ids": package.source_ids},
                prepare=lambda d: apply_deterministic_rules(d, package),
            )
        except LLMError as exc:
            raise AgentError(f"Verification Agent failed: {exc}") from None

        report = VerificationReport.from_checks(response.claims, len(package.sources),
                                                STANDARD_LIMITATIONS + response.limitations, mode="live")
        logger.info("Verification: %d claims, status=%s", report.claims_checked, report.overall_status)
        return report
