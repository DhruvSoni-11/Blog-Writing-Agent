"""End-to-end LangGraph workflow tests with mocked LLM and search."""

from app.agents.base import AgentContext
from app.config import load_settings
from app.demo.fixtures import DEMO_BANNER, FixtureSearchProvider
from app.graph import recursion_limit_for, run_workflow
from app.tools.web_search import ResearchService
from tests.conftest import BAD_BODY, CLAIMS, standard_replies, writer_reply


def test_live_workflow_completes_with_one_correction_pass(live_context):
    ctx, llm = live_context()
    events = []
    ctx.progress = lambda stage, event, msg: events.append((stage, event))
    state = run_workflow(ctx, "How Agentic AI Works", audience="students", tone="friendly", word_count=400)

    assert state["execution_status"] in ("completed", "completed_with_warnings")
    assert state["retry_count"] == 1
    assert llm.count("writer.draft") == 1 and llm.count("writer.revision") == 1
    # The research process is not repeated for the correction pass.
    assert llm.count("research.findings") == 1 and llm.count("verification.claims") == 1

    blog = state["final_blog"]
    assert "35%" not in blog.markdown
    retrieved = {s.url for s in state["research_package"].sources}
    reference_lines = blog.markdown.split("## References")[1]
    assert all(url in reference_lines for url in (s.url for s in state["research_package"].sources
                                                  if s.source_id in blog.cited_source_ids))
    for line in reference_lines.splitlines():
        if line.startswith("- **["):
            assert any(url in line for url in retrieved)
    assert "DEMONSTRATION" not in blog.markdown

    meta = state["execution_metadata"]
    assert meta["llm_model"] == "fake/test-model:free"
    assert meta["search_provider"] == "fake-search"
    assert meta["agent_status"]["writer"] == "completed (correction pass 1)"
    assert meta["elapsed_seconds"] >= 0 and meta["workflow_end"]
    assert ("research", "done") in events and ("validation", "done") in events

    # The revision prompt carried the verification feedback.
    revision = next(c for c in llm.calls if c["purpose"] == "writer.revision")
    assert "CORRECTION PASS" in revision["user"] and "C2" in revision["user"]


def test_zero_retries_never_revises_and_removes_unsupported_sentence(live_context):
    ctx, llm = live_context(VERIFICATION_RETRIES=0)
    state = run_workflow(ctx, "How Agentic AI Works", word_count=400)
    assert llm.count("writer.revision") == 0
    assert state["retry_count"] == 0
    assert "35%" not in state["final_blog"].markdown
    assert state["execution_status"] == "completed_with_warnings"
    assert any("Removed 1 sentence" in w for w in state["warnings"])


def test_retry_limit_prevents_infinite_loop(live_context):
    replies = standard_replies()
    replies["writer.revision"] = writer_reply(BAD_BODY)  # the writer never fixes the problem
    for retries in (1, 3):
        ctx, llm = live_context(replies=dict(replies), VERIFICATION_RETRIES=retries)
        state = run_workflow(ctx, "How Agentic AI Works", word_count=400)
        assert llm.count("writer.revision") == retries
        assert state["retry_count"] == retries
        assert state["final_blog"] is not None
        assert any("Unresolved after correction pass" in w for w in state["warnings"])
    assert recursion_limit_for(3) > 6 + 2 * 3


def test_no_search_results_stops_workflow(live_context):
    ctx, llm = live_context(results=[])
    state = run_workflow(ctx, "Nothing to find", word_count=400)
    assert state["execution_status"] == "failed"
    assert state["final_blog"] is None and state["research_package"] is None
    assert "Web research failed" in state["errors"][0]
    status = state["execution_metadata"]["agent_status"]
    assert status["research"] == "failed" and status["summary"] == "skipped" and status["writer"] == "skipped"
    assert llm.count("summary.brief") == 0


def test_insufficient_evidence_stops_before_writing(live_context):
    replies = standard_replies()
    replies["verification.claims"] = {"claims": [
        {"claim_text": c["claim_text"], "status": "unverifiable", "supporting_source_ids": [],
         "explanation": "not addressed"} for c in CLAIMS["claims"]]}
    ctx, llm = live_context(replies=replies)
    state = run_workflow(ctx, "How Agentic AI Works", word_count=400)
    assert state["execution_status"] == "failed"
    assert llm.count("writer.draft") == 0
    assert state["execution_metadata"]["agent_status"]["verification"] == "insufficient_evidence"


def test_llm_failure_is_reported_not_replaced_with_mock_data(live_context):
    from app.llm_client import LLMAuthError
    replies = standard_replies()
    replies["summary.brief"] = LLMAuthError("OpenRouter rejected the API key (401).")
    ctx, _ = live_context(replies=replies)
    state = run_workflow(ctx, "How Agentic AI Works", word_count=400)
    assert state["execution_status"] == "failed"
    assert "401" in state["errors"][0]
    assert state["research_brief"] is None and state["final_blog"] is None


def test_demo_mode_runs_without_api_keys_or_network():
    settings = load_settings(env={})  # no keys at all
    assert not settings.llm_configured
    topic = "Quantum Computing Basics"
    service = ResearchService([FixtureSearchProvider(topic)], max_results=8, delay_seconds=0, simulated=True)
    ctx = AgentContext(settings=settings, research_service=service, llm=None, demo=True)
    state = run_workflow(ctx, topic)  # the autouse fixture blocks any network access

    assert state["execution_status"] == "completed"
    package = state["research_package"]
    assert package.mode == "demonstration" and package.search_provider.endswith("(simulated)")
    assert all(s.simulated and s.url.endswith(".invalid/simulated-source") for s in package.sources)
    assert state["verification_report"].mode == "demonstration"
    assert state["retry_count"] == 1  # the demo deliberately exercises the correction loop
    blog = state["final_blog"].markdown
    assert DEMO_BANNER in blog and "SIMULATED" in blog
    assert "40 percent" not in blog
    assert state["execution_metadata"]["mode"] == "demonstration"
