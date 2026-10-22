"""Shared runtime context passed to every agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from app.config import Settings
from app.llm_client import LLMClient
from app.tools.web_search import ResearchService

ProgressCallback = Callable[[str, str, str], None]  # (stage, event, message)


class AgentError(Exception):
    """An agent could not produce a valid output. The message is safe to show to users."""


def _no_progress(stage: str, event: str, message: str) -> None:
    return None


@dataclass
class AgentContext:
    settings: Settings
    research_service: ResearchService
    llm: LLMClient | None = None
    demo: bool = False
    progress: ProgressCallback = field(default=_no_progress)

    @property
    def mode(self) -> str:
        return "demonstration" if self.demo else "live"

    @property
    def model_name(self) -> str:
        from app.demo.fixtures import DEMO_MODEL

        if self.demo:
            return DEMO_MODEL
        return self.llm.model_name if self.llm else "not configured"

    def require_llm(self) -> LLMClient:
        """Live mode never falls back to templates: no LLM means an explicit error."""
        if self.llm is None:
            raise AgentError("Live mode requires an OpenRouter model. Configure .env or run with --demo.")
        return self.llm
