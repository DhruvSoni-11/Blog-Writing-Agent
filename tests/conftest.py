"""Shared fixtures: a scripted fake LLM and fake search providers. No network, no API keys."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.base import AgentContext  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.tools.web_search import RawResult, ResearchService  # noqa: E402


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if any test tries to open a real network connection."""
    def guard(*args, **kwargs):
        raise RuntimeError("Network access is disabled in tests")
    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


class FakeLLM:
    """Returns scripted replies keyed by ``purpose``. A list is consumed in order (last one repeats)."""

    model_name = "fake/test-model:free"

    def __init__(self, replies: dict):
        self.replies = replies
        self.calls: list[dict] = []

    def complete(self, system, user, *, purpose, max_tokens=2048, temperature=0.2):
        self.calls.append({"purpose": purpose, "system": system, "user": user})
        reply = self.replies.get(purpose)
        if reply is None and "." in purpose:
            reply = self.replies.get(purpose.rsplit(".", 1)[0])  # e.g. "x.repair" -> "x"
        if reply is None:
            raise AssertionError(f"FakeLLM has no reply for purpose {purpose!r}")
        if isinstance(reply, list):
            count = sum(1 for c in self.calls if c["purpose"] == purpose)
            reply = reply[min(count - 1, len(reply) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)

    def count(self, purpose: str) -> int:
        return sum(1 for c in self.calls if c["purpose"] == purpose)


class FakeProvider:
    def __init__(self, name="fake-search", results=None, error: Exception | None = None):
        self.name = name
        self.results = results or []
        self.error = error
        self.queries: list[str] = []

    def search(self, query, max_results):
        self.queries.append(query)
        if self.error:
            raise self.error
        return list(self.results)


SEARCH_RESULTS = [
    RawResult("Agentic AI overview", "https://en.wikipedia.org/wiki/Agentic_AI",
              "Agentic AI systems plan and take actions toward goals using tools.", None),
    RawResult("Agent architectures survey", "https://arxiv.org/abs/2401.00001",
              "A survey of LLM-based agents describing planning, memory and tool use components.", "2024-01-02"),
    RawResult("Building agents - documentation", "https://docs.example.org/agents/intro",
              "Agents call tools in a loop and stop when the task is complete.", None),
    RawResult("AI risk management framework", "https://www.nist.gov/itl/ai-risk-management-framework",
              "The framework recommends human oversight and evaluation of AI systems.", "2023-01-26"),
    RawResult("Duplicate of overview", "https://en.wikipedia.org/wiki/Agentic_AI/?utm_source=x", "dup", None),
    RawResult("Broken", "not a url", "bad", None),
    RawResult("FTP", "ftp://files.example.org/x", "bad scheme", None),
    RawResult("Forum thread", "https://www.reddit.com/r/ai/comments/1", "People discuss agents.", None),
]

QUERIES = {"queries": ["agentic ai overview", "agentic ai architecture", "agentic ai risks"]}

FINDINGS = {
    "findings": [
        {"statement": "Agentic AI systems plan and act toward goals using tools.", "category": "key_fact",
         "basis": "retrieved_evidence", "source_ids": ["S1", "S3"]},
        {"statement": "Agents combine planning, memory and tool use.", "category": "concept",
         "basis": "retrieved_evidence", "source_ids": ["S2"]},
        {"statement": "Human oversight is recommended.", "category": "key_fact",
         "basis": "retrieved_evidence", "source_ids": ["S4"]},
        {"statement": "Invented finding citing a fake source.", "category": "key_fact",
         "basis": "retrieved_evidence", "source_ids": ["S99"]},
    ],
    "notes": ["Snippets are short."],
}

BRIEF = {
    "topic": "ignored - replaced by the real topic",
    "executive_summary": "Agentic AI systems plan and act with tools; oversight matters.",
    "key_points": ["Agents plan and act", "Components: planning, memory, tools", "Oversight is recommended"],
    "supporting_evidence": [
        {"point": "Agents plan and take actions using tools.", "source_ids": ["S1", "S3"]},
        {"point": "Survey describes planning, memory and tools.", "source_ids": ["S2"]},
        {"point": "Fabricated evidence.", "source_ids": ["S42"]},
    ],
    "important_terms": [{"term": "Tool use", "definition": "Calling external functions."}, "Memory: stored context"],
    "examples": ["A coding agent that runs tests."],
    "disagreements": [],
    "limitations": ["Snippet-level evidence only."],
    "recommended_blog_angle": "Practical introduction for developers.",
}

CLAIMS = {
    "claims": [
        {"claim_text": "Agentic AI systems plan and take actions using tools.", "status": "supported",
         "supporting_source_ids": ["S1"], "explanation": "S1 states this.", "recommended_correction": None},
        {"claim_text": "Agentic AI reduces software errors by 35% in production teams.", "status": "supported",
         "supporting_source_ids": ["S2"], "explanation": "Model believed it.", "recommended_correction": None},
        {"claim_text": "Agents combine planning, memory and tool use.", "status": "SUPPORTED",
         "supporting_source_ids": ["S2"], "explanation": "S2 lists these.", "recommended_correction": None},
        {"claim_text": "Most companies will deploy agents next year.", "status": "supported",
         "supporting_source_ids": [], "explanation": "No source.", "recommended_correction": None},
        {"claim_text": "Human oversight is recommended.", "status": "probably-true",
         "supporting_source_ids": ["S4"], "explanation": "Unknown status label.", "recommended_correction": None},
    ],
    "limitations": [],
}

GOOD_BODY = """## Introduction

Agentic AI systems plan and take actions toward goals using tools [S1].

## How Agents Work

### Components

A survey describes planning, memory and tool use as core components [S2]. Agents call tools in a loop [S3].

## Risks and Oversight

Frameworks recommend human oversight and evaluation [S4].

## Practical Implications

Start with small, well-scoped tasks and review the agent's actions.

## Conclusion

Agentic AI is a loop of planning and acting; keep humans involved [S1].
"""

BAD_BODY = GOOD_BODY.replace(
    "## Risks and Oversight",
    "Agentic AI reduces software errors by 35% in production teams.\n\n## Risks and Oversight",
)


def writer_reply(body: str, title: str = "How Agentic AI Works: A Practical Guide") -> str:
    return f"TITLE: {title}\nDESCRIPTION: A short guide.\nKEYWORDS: agentic ai, agents, tools\n---\n{body}"


def standard_replies(writer=None) -> dict:
    return {
        "research.queries": QUERIES,
        "research.findings": FINDINGS,
        "summary.brief": BRIEF,
        "verification.claims": CLAIMS,
        "writer.draft": writer or writer_reply(BAD_BODY),
        "writer.revision": writer_reply(GOOD_BODY),
    }


@pytest.fixture
def make_settings():
    def factory(**overrides):
        env = {"OPENROUTER_API_KEY": "sk-or-test-key-123456", "OPENROUTER_MODEL": "fake/test-model:free",
               "VERIFICATION_RETRIES": "1", "MAX_SEARCH_RESULTS": "8", "MAX_RESEARCH_QUERIES": "3"}
        env.update({k: str(v) for k, v in overrides.items()})
        return load_settings(env={k: v for k, v in env.items() if v != "None"})
    return factory


@pytest.fixture
def live_context(make_settings):
    def factory(replies=None, results=None, providers=None, **settings_overrides):
        settings = make_settings(**settings_overrides)
        service = ResearchService(providers or [FakeProvider(results=SEARCH_RESULTS if results is None else results)],
                                  max_results=settings.max_search_results, delay_seconds=0)
        llm = FakeLLM(replies or standard_replies())
        return AgentContext(settings=settings, research_service=service, llm=llm, demo=False), llm
    return factory
