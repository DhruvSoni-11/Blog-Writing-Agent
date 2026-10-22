"""Deterministic LangGraph orchestration of the four agents.

START -> research -> summary -> verification -> writer -> validation -> finalize -> END
                                     |                      |
                                     | (insufficient)       | (material issues and retries left)
                                     v                      v
                                    END                   writer   (bounded correction pass)

Every node stops the run on failure. The correction loop is bounded by VERIFICATION_RETRIES and
LangGraph's recursion limit is set as a second safety net.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone

from importlib.metadata import PackageNotFoundError, version

from langgraph.graph import END, START, StateGraph

from app.agents.base import AgentContext, AgentError
from app.agents.blog_writer_agent import BlogWriterAgent
from app.agents.research_agent import ResearchAgent
from app.agents.summary_agent import SummaryAgent
from app.agents.verification_agent import VerificationAgent
from app.demo.fixtures import DEMO_BANNER
from app.llm_client import LLMError
from app.state import BlogState, initial_state
from app.tools.web_search import SearchError
from app.validation import assemble_final_blog, validate_draft

logger = logging.getLogger(__name__)

STAGES = {
    "research": "Research Agent",
    "summary": "Summary Agent",
    "verification": "Verification Agent",
    "writer": "Blog Writer Agent",
    "validation": "Output validation",
    "finalize": "Final assembly",
}


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, (AgentError, LLMError, SearchError)):
        return str(exc)
    return f"Unexpected {type(exc).__name__}: {str(exc)[:300]}"


def build_graph(ctx: AgentContext):
    research_agent = ResearchAgent(ctx)
    summary_agent = SummaryAgent(ctx)
    verification_agent = VerificationAgent(ctx)
    writer_agent = BlogWriterAgent(ctx)
    max_retries = ctx.settings.verification_retries

    def node(stage: str, work: Callable[[BlogState], dict]) -> Callable[[BlogState], dict]:
        label = STAGES[stage]

        def run(state: BlogState) -> dict:
            meta = copy.deepcopy(state["execution_metadata"])
            ctx.progress(stage, "start", label)
            started = time.perf_counter()
            try:
                update = work(state)
            except Exception as exc:  # each node converts failures into state, never crashes the CLI
                logger.exception("%s failed", label)
                message = f"{label}: {_safe_error(exc)}"
                meta["agent_status"][stage] = "failed"
                ctx.progress(stage, "failed", message)
                return {"errors": state["errors"] + [message], "execution_status": "failed",
                        "execution_metadata": meta}
            meta["agent_status"][stage] = update.pop("_status", "completed")
            meta["stage_seconds"][stage] = round(time.perf_counter() - started, 2)
            summary_text = update.pop("_summary", label)
            update["execution_metadata"] = meta
            if update.get("execution_status") == "failed":
                ctx.progress(stage, "failed", (update.get("errors") or [label])[-1])
            else:
                ctx.progress(stage, "done", summary_text)
            return update
        return run

    # ------------------------------------------------------------------ nodes
    def research(state: BlogState) -> dict:
        package = research_agent.run(state["topic"], state.get("audience"))
        meta_warnings = state["warnings"] + package.notes
        return {"research_package": package, "warnings": meta_warnings,
                "_summary": f"{len(package.sources)} sources via {package.search_provider} "
                            f"(research {package.research_status})"}

    def summary(state: BlogState) -> dict:
        brief = summary_agent.run(state["research_package"])
        return {"research_brief": brief,
                "_summary": f"{len(brief.key_points)} key points, {len(brief.supporting_evidence)} evidence items"}

    def verification(state: BlogState) -> dict:
        report = verification_agent.run(state["research_package"], state["research_brief"])
        update = {"verification_report": report,
                  "_summary": f"{report.claims_checked} claims checked, status {report.overall_status}"}
        if report.overall_status == "insufficient_evidence":
            update["errors"] = state["errors"] + [
                "Verification Agent: the retrieved evidence does not support the brief's claims, so a "
                "responsible blog cannot be written. Try a more specific topic or a different search provider."
            ]
            update["execution_status"] = "failed"
            update["_status"] = "insufficient_evidence"
        return update

    def writer(state: BlogState) -> dict:
        validation = state.get("validation")
        correction = validation is not None and not validation.passed
        retry_count = state["retry_count"] + (1 if correction else 0)
        feedback = None
        if correction:
            report = state["verification_report"]
            feedback = validation.material_issues + [f"Verification correction - {c}" for c in report.corrections]
        draft = writer_agent.run(state["research_package"], state["research_brief"], state["verification_report"],
                                 state["requested_word_count"], state.get("audience"), state.get("tone"),
                                 feedback=feedback, previous_draft=state.get("draft") if correction else None)
        return {"draft": draft, "retry_count": retry_count, "validation": None,
                "_status": f"completed (correction pass {retry_count})" if correction else "completed",
                "_summary": f"correction pass {retry_count} written" if correction else f"draft '{draft.title}'"}

    def validation(state: BlogState) -> dict:
        result = validate_draft(state["draft"], state["research_package"], state["verification_report"],
                                state["requested_word_count"])
        summary_text = ("passed" if result.passed else f"{len(result.material_issues)} material issue(s)")
        return {"validation": result, "_summary": f"{summary_text}, {len(result.warnings)} warning(s)"}

    def finalize(state: BlogState) -> dict:
        final = assemble_final_blog(state["draft"], state["validation"], state["research_package"],
                                    state["verification_report"], ctx.model_name,
                                    demo_banner=DEMO_BANNER if ctx.demo else None)
        warnings = state["warnings"] + final.warnings
        status = "completed_with_warnings" if warnings else "completed"
        return {"final_blog": final, "warnings": warnings, "execution_status": status,
                "_summary": f"{final.word_count} words, {len(final.cited_source_ids)} sources cited"}

    # ------------------------------------------------------------------ routing
    def continue_or_stop(next_node: str):
        def route(state: BlogState) -> str:
            return "stop" if state.get("execution_status") == "failed" else next_node
        return route

    def route_after_validation(state: BlogState) -> str:
        if state.get("execution_status") == "failed":
            return "stop"
        result = state["validation"]
        if not result.passed and state["retry_count"] < max_retries:
            logger.info("Routing back to writer for correction (retry %d of %d)", state["retry_count"] + 1, max_retries)
            return "revise"
        return "finalize"

    graph = StateGraph(BlogState)
    for stage, fn in [("research", research), ("summary", summary), ("verification", verification),
                      ("writer", writer), ("validation", validation), ("finalize", finalize)]:
        graph.add_node(stage, node(stage, fn))

    graph.add_edge(START, "research")
    graph.add_conditional_edges("research", continue_or_stop("summary"), {"summary": "summary", "stop": END})
    graph.add_conditional_edges("summary", continue_or_stop("verification"), {"verification": "verification", "stop": END})
    graph.add_conditional_edges("verification", continue_or_stop("writer"), {"writer": "writer", "stop": END})
    graph.add_conditional_edges("writer", continue_or_stop("validation"), {"validation": "validation", "stop": END})
    graph.add_conditional_edges("validation", route_after_validation,
                                {"revise": "writer", "finalize": "finalize", "stop": END})
    graph.add_edge("finalize", END)
    return graph.compile()


def _langgraph_version() -> str:
    try:
        return version("langgraph")
    except PackageNotFoundError:
        return "unknown"


def recursion_limit_for(retries: int) -> int:
    """Steps needed for the longest legal path, plus headroom; LangGraph aborts beyond this."""
    return 6 + 2 * retries + 4


def run_workflow(ctx: AgentContext, topic: str, audience: str | None = None, tone: str | None = None,
                 word_count: int | None = None) -> BlogState:
    settings = ctx.settings
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    metadata = {
        "workflow_start": started.isoformat(timespec="seconds"),
        "mode": ctx.mode,
        "llm_model": ctx.model_name,
        "search_provider": None,
        "agent_status": {stage: "pending" for stage in STAGES},
        "stage_seconds": {},
        "retry_count": 0,
        "max_retries": settings.verification_retries,
        "langgraph_version": _langgraph_version(),
        "settings": {
            "max_search_results": settings.max_search_results,
            "max_research_queries": settings.max_research_queries,
            "tavily_configured": settings.tavily_configured,
        },
    }
    state = initial_state(topic.strip(), audience, tone, word_count or settings.blog_word_count, metadata)
    app = build_graph(ctx)
    try:
        final: BlogState = app.invoke(state, config={"recursion_limit": recursion_limit_for(settings.verification_retries)})
    except Exception as exc:  # e.g. GraphRecursionError - should be impossible, but never loop forever
        logger.exception("Workflow aborted")
        final = {**state, "errors": state["errors"] + [f"Workflow aborted: {_safe_error(exc)}"],
                 "execution_status": "failed"}

    finished = datetime.now(timezone.utc)
    meta = final["execution_metadata"]
    package = final.get("research_package")
    meta.update({
        "workflow_end": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.perf_counter() - clock, 2),
        "search_provider": package.search_provider if package else None,
        "retry_count": final.get("retry_count", 0),
        "execution_status": final.get("execution_status"),
        "errors": final.get("errors", []),
        "warnings": final.get("warnings", []),
    })
    for stage, status in meta["agent_status"].items():
        if status == "pending" and final.get("execution_status") == "failed":
            meta["agent_status"][stage] = "skipped"
    return final
