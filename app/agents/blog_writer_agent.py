"""Agent 4 - Blog Writer Agent: writes the Markdown post, and revises it once on feedback."""

from __future__ import annotations

import logging
import re

from app.agents.base import AgentContext, AgentError
from app.demo.fixtures import demo_blog_markdown
from app.llm_client import LLMError, LLMResponseError
from app.models import BlogDraft, ResearchBrief, ResearchPackage, VerificationReport
from app.prompts import WRITER_SYSTEM, writer_prompt

logger = logging.getLogger(__name__)

_META_RE = re.compile(r"^\s*\**\s*(TITLE|DESCRIPTION|KEYWORDS)\s*\**\s*:\s*\**\s*(.*?)\s*\**\s*$", re.IGNORECASE)


def parse_writer_output(text: str) -> BlogDraft:
    """Parse the ``TITLE/DESCRIPTION/KEYWORDS/---/body`` format. Raises ValueError if unusable."""
    text = re.sub(r"^\s*```(?:markdown|md)?\s*\n|\n\s*```\s*$", "", (text or "").strip(), flags=re.IGNORECASE)
    meta: dict[str, str] = {}
    lines = text.splitlines()
    body_start = 0
    for index, line in enumerate(lines):
        match = _META_RE.match(line)
        if match:
            meta[match.group(1).lower()] = match.group(2).strip()
            body_start = index + 1
        elif line.strip() == "---" and meta:
            body_start = index + 1
            break
        elif line.strip() and meta and line.lstrip().startswith("#"):
            body_start = index
            break
    body = "\n".join(lines[body_start:]).strip()

    if "title" not in meta:  # tolerate a leading "# Title" heading instead of TITLE:
        h1 = re.match(r"^#\s+(.+)$", body, flags=re.MULTILINE)
        if h1:
            meta["title"] = h1.group(1).strip()
    body = re.sub(r"^#\s+.+\n+", "", body, count=1) if body.startswith("# ") else body
    if not meta.get("title") or "## " not in body:
        raise ValueError("missing TITLE or markdown sections")

    keywords = [k.strip(" #*") for k in meta.get("keywords", "").split(",") if k.strip(" #*")]
    description = meta.get("description") or meta["title"]
    return BlogDraft(title=meta["title"].strip('"*# ')[:200], description=description[:500],
                     keywords=keywords[:8], body_markdown=body)


class BlogWriterAgent:
    name = "Blog Writer Agent"

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    def run(self, package: ResearchPackage, brief: ResearchBrief, report: VerificationReport,
            word_count: int, audience: str | None = None, tone: str | None = None,
            feedback: list[str] | None = None, previous_draft: BlogDraft | None = None) -> BlogDraft:
        if self.ctx.demo:
            return self._demo_draft(package, brief, correction_pass=bool(feedback))

        llm = self.ctx.require_llm()
        prompt = writer_prompt(package, brief, report, word_count, audience, tone, feedback,
                               previous_draft.body_markdown if previous_draft else None)
        max_tokens = min(12000, int(word_count * 2.2) + 3000)  # headroom for reasoning models
        purpose = "writer.revision" if feedback else "writer.draft"
        try:
            reply = llm.complete(WRITER_SYSTEM, prompt, purpose=purpose, max_tokens=max_tokens, temperature=0.5)
            try:
                return parse_writer_output(reply)
            except ValueError as exc:  # one bounded format-repair attempt
                logger.warning("Writer output unparseable (%s); requesting reformat", exc)
                repair = prompt + (
                    "\n\nYour previous reply did not follow the output format. Start with the lines "
                    "TITLE:, DESCRIPTION:, KEYWORDS:, then a line '---', then the Markdown body using ## headings."
                )
                reply = llm.complete(WRITER_SYSTEM, repair, purpose=f"{purpose}.repair",
                                     max_tokens=max_tokens, temperature=0.3)
                try:
                    return parse_writer_output(reply)
                except ValueError:
                    raise LLMResponseError("the model did not return a post in the required format") from None
        except LLMError as exc:
            raise AgentError(f"Blog Writer Agent failed: {exc}") from None

    def _demo_draft(self, package: ResearchPackage, brief: ResearchBrief, correction_pass: bool) -> BlogDraft:
        topic = package.topic
        return BlogDraft(
            title=f"{topic}: A Beginner's Guide (Demonstration)",
            description=f"A simulated demonstration post about {topic}, generated without live research.",
            keywords=[topic.lower(), "demonstration", "blogforge", "multi-agent pipeline"],
            body_markdown=demo_blog_markdown(topic, brief, include_unsupported_claim=not correction_pass),
        )
