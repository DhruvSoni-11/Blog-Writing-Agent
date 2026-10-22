"""Typed shared state for the LangGraph workflow."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from app.models import BlogDraft, FinalBlog, ResearchBrief, ResearchPackage, ValidationResult, VerificationReport

ExecutionStatus = Literal["running", "completed", "completed_with_warnings", "failed"]


class BlogState(TypedDict, total=False):
    # Inputs
    topic: str
    audience: str | None
    tone: str | None
    requested_word_count: int
    # Agent outputs
    research_package: ResearchPackage | None
    research_brief: ResearchBrief | None
    verification_report: VerificationReport | None
    draft: BlogDraft | None
    validation: ValidationResult | None
    final_blog: FinalBlog | None
    # Bookkeeping (nodes return complete lists, so no reducers are needed)
    retry_count: int
    errors: list[str]
    warnings: list[str]
    execution_metadata: dict[str, Any]
    execution_status: ExecutionStatus


def initial_state(topic: str, audience: str | None, tone: str | None, word_count: int,
                  metadata: dict[str, Any]) -> BlogState:
    return BlogState(
        topic=topic, audience=audience, tone=tone, requested_word_count=word_count,
        research_package=None, research_brief=None, verification_report=None, draft=None,
        validation=None, final_blog=None, retry_count=0, errors=[], warnings=[],
        execution_metadata=metadata, execution_status="running",
    )
