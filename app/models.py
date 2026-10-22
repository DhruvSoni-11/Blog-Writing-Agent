"""Pydantic schemas for every structured artefact passed between agents.

Source-ID references can be checked against the retrieved sources by passing
``context={"valid_source_ids": {...}}`` to ``model_validate``. ResearchPackage checks its own
findings against its own sources automatically.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from app.utils.text_utils import is_valid_url

VerificationStatus = Literal["supported", "partially_supported", "unsupported", "contradictory", "unverifiable"]
RunMode = Literal["live", "demonstration"]

SOURCE_ID_PATTERN = r"^S\d+$"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _check_ids(ids: list[str], info: ValidationInfo) -> list[str]:
    valid = (info.context or {}).get("valid_source_ids")
    if valid is not None:
        unknown = [i for i in ids if i not in valid]
        if unknown:
            raise ValueError(f"unknown source IDs: {', '.join(unknown)}")
    return ids


class _Strict(BaseModel):
    model_config = {"extra": "ignore", "str_strip_whitespace": True}


# ============================================================== Research Agent
class Source(_Strict):
    source_id: str = Field(pattern=SOURCE_ID_PATTERN)
    title: str = Field(min_length=1)
    url: str
    snippet: str = ""
    publication_date: str | None = None
    retrieved_at: str = Field(default_factory=utc_now_iso)
    provider: str
    source_type: str = "general"
    simulated: bool = False

    @field_validator("url")
    @classmethod
    def _url_must_be_valid(cls, value: str) -> str:
        if not is_valid_url(value):
            raise ValueError(f"invalid URL: {value!r}")
        return value


class Finding(_Strict):
    finding_id: str = Field(pattern=r"^F\d+$")
    statement: str = Field(min_length=1)
    category: Literal["key_fact", "concept", "example", "disagreement"] = "key_fact"
    # "retrieved_evidence" = stated by a source; "model_interpretation" = the model's synthesis.
    basis: Literal["retrieved_evidence", "model_interpretation"] = "retrieved_evidence"
    source_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _evidence_needs_sources(self) -> Finding:
        if self.basis == "retrieved_evidence" and not self.source_ids:
            raise ValueError("retrieved_evidence findings must cite at least one source ID")
        return self


class ResearchPackage(_Strict):
    topic: str = Field(min_length=1)
    audience: str | None = None
    research_queries: list[str] = Field(min_length=1)
    sources: list[Source]
    findings: list[Finding] = Field(default_factory=list)
    search_provider: str
    research_status: Literal["sufficient", "limited", "insufficient"]
    mode: RunMode = "live"
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent_ids(self) -> ResearchPackage:
        ids = [s.source_id for s in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate source IDs")
        known = set(ids)
        for finding in self.findings:
            unknown = [i for i in finding.source_ids if i not in known]
            if unknown:
                raise ValueError(f"finding {finding.finding_id} cites unknown sources: {unknown}")
        return self

    @property
    def source_ids(self) -> set[str]:
        return {s.source_id for s in self.sources}

    def get_source(self, source_id: str) -> Source | None:
        return next((s for s in self.sources if s.source_id == source_id), None)


# ============================================================== Summary Agent
class EvidenceItem(_Strict):
    point: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)

    @field_validator("source_ids")
    @classmethod
    def _known_ids(cls, value: list[str], info: ValidationInfo) -> list[str]:
        return _check_ids(value, info)


class TermDefinition(_Strict):
    term: str = Field(min_length=1)
    definition: str = Field(min_length=1)


class ResearchBrief(_Strict):
    topic: str = Field(min_length=1)
    executive_summary: str = Field(min_length=1)
    key_points: list[str] = Field(min_length=1)
    supporting_evidence: list[EvidenceItem] = Field(min_length=1)
    important_terms: list[TermDefinition] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommended_blog_angle: str = Field(min_length=1)


# ============================================================== Verification Agent
class ClaimCheck(_Strict):
    claim_id: str = Field(pattern=r"^C\d+$")
    claim_text: str = Field(min_length=1)
    status: VerificationStatus
    supporting_source_ids: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1)
    recommended_correction: str | None = None

    @field_validator("supporting_source_ids")
    @classmethod
    def _known_ids(cls, value: list[str], info: ValidationInfo) -> list[str]:
        return _check_ids(value, info)

    @model_validator(mode="after")
    def _support_needs_sources(self) -> ClaimCheck:
        if self.status in ("supported", "partially_supported") and not self.supporting_source_ids:
            raise ValueError(f"claim {self.claim_id} is '{self.status}' but cites no sources")
        return self


COVERAGE_METHOD = (
    "claim_support_ratio = (supported + partially_supported claims) / claims_checked; "
    "source_utilization_ratio = distinct retrieved sources cited by any supported or partially "
    "supported claim / total retrieved sources. Ratios are 0 when the denominator is 0."
)


class SourceCoverage(_Strict):
    claims_checked: int = Field(ge=0)
    claims_with_support: int = Field(ge=0)
    claim_support_ratio: float = Field(ge=0, le=1)
    sources_total: int = Field(ge=0)
    sources_used: int = Field(ge=0)
    source_utilization_ratio: float = Field(ge=0, le=1)
    method: str = COVERAGE_METHOD


class VerificationReport(_Strict):
    overall_status: Literal["pass", "pass_with_warnings", "needs_revision", "insufficient_evidence"]
    claims_checked: int = Field(ge=0)
    verified_claims: list[ClaimCheck] = Field(default_factory=list)
    partially_supported_claims: list[ClaimCheck] = Field(default_factory=list)
    unsupported_claims: list[ClaimCheck] = Field(default_factory=list)
    contradictory_claims: list[ClaimCheck] = Field(default_factory=list)
    unverifiable_claims: list[ClaimCheck] = Field(default_factory=list)
    corrections: list[str] = Field(default_factory=list)
    source_coverage: SourceCoverage
    limitations: list[str] = Field(default_factory=list)
    mode: RunMode = "live"

    @model_validator(mode="after")
    def _counts_match(self) -> VerificationReport:
        if self.claims_checked != len(self.all_claims):
            raise ValueError("claims_checked does not match the number of claims in the report")
        return self

    @property
    def all_claims(self) -> list[ClaimCheck]:
        return (
            self.verified_claims + self.partially_supported_claims + self.unsupported_claims
            + self.contradictory_claims + self.unverifiable_claims
        )

    @property
    def material_issues(self) -> list[ClaimCheck]:
        """Claims that must not appear in the blog as stated."""
        return self.unsupported_claims + self.contradictory_claims

    @classmethod
    def from_checks(
        cls, checks: list[ClaimCheck], total_sources: int, limitations: list[str], mode: RunMode = "live"
    ) -> VerificationReport:
        """Deterministically group claims, compute coverage and the overall status."""
        by_status: dict[str, list[ClaimCheck]] = {s: [] for s in VerificationStatus.__args__}
        for check in checks:
            by_status[check.status].append(check)
        supported = by_status["supported"] + by_status["partially_supported"]
        used = {sid for c in supported for sid in c.supporting_source_ids}
        n = len(checks)

        if n == 0 or not supported:
            status = "insufficient_evidence"
        elif by_status["unsupported"] or by_status["contradictory"]:
            status = "needs_revision"
        elif by_status["partially_supported"] or by_status["unverifiable"]:
            status = "pass_with_warnings"
        else:
            status = "pass"

        return cls(
            overall_status=status,
            claims_checked=n,
            verified_claims=by_status["supported"],
            partially_supported_claims=by_status["partially_supported"],
            unsupported_claims=by_status["unsupported"],
            contradictory_claims=by_status["contradictory"],
            unverifiable_claims=by_status["unverifiable"],
            corrections=[
                f"{c.claim_id}: {c.recommended_correction}"
                for c in checks
                if c.recommended_correction and c.status != "supported"
            ],
            source_coverage=SourceCoverage(
                claims_checked=n,
                claims_with_support=len(supported),
                claim_support_ratio=round(len(supported) / n, 3) if n else 0.0,
                sources_total=total_sources,
                sources_used=len(used),
                source_utilization_ratio=round(len(used) / total_sources, 3) if total_sources else 0.0,
            ),
            limitations=limitations,
            mode=mode,
        )


# ============================================================== Blog Writer Agent
class BlogDraft(_Strict):
    title: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=1, max_length=500)
    keywords: list[str] = Field(default_factory=list, max_length=12)
    body_markdown: str = Field(min_length=1)


class ValidationResult(_Strict):
    material_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    cited_source_ids: list[str] = Field(default_factory=list)
    flagged_sentences: list[str] = Field(default_factory=list)
    word_count: int = 0

    @property
    def passed(self) -> bool:
        return not self.material_issues


class FinalBlog(_Strict):
    title: str
    description: str
    keywords: list[str]
    markdown: str
    word_count: int
    cited_source_ids: list[str]
    warnings: list[str] = Field(default_factory=list)
