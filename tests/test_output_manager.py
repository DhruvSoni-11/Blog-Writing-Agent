import json
from datetime import date

import pytest

from app.agents.base import AgentContext
from app.config import load_settings
from app.demo.fixtures import FixtureSearchProvider
from app.graph import run_workflow
from app.output_manager import OutputManager
from app.tools.web_search import ResearchService
from app.utils.text_utils import slugify

TODAY = date(2026, 10, 15)


@pytest.mark.parametrize("topic,expected", [
    ("How Agentic AI Works", "how-agentic-ai-works"),
    ("../../etc/passwd", "etc-passwd"),
    ('C:\\Windows\\system32 <script>"|?*', "c-windows-system32-script"),
    ("Café & Crème Brûlée!", "cafe-creme-brulee"),
    ("CON", "topic-con"),
    ("???", "untitled"),
    ("", "untitled"),
])
def test_safe_filename_generation(topic, expected):
    assert slugify(topic) == expected


def test_slug_length_is_limited():
    assert len(slugify("word " * 100)) <= 50


def test_output_directory_is_created_and_unique(tmp_path):
    manager = OutputManager(tmp_path / "nested" / "output")
    first = manager.create_run_dir("Agentic AI", TODAY)
    second = manager.create_run_dir("Agentic AI", TODAY)
    assert first.name == "2026-10-15-agentic-ai"
    assert second.name == "2026-10-15-agentic-ai-2"
    assert first.is_dir() and second.is_dir()


def test_refuses_to_write_outside_output_dir(tmp_path):
    manager = OutputManager(tmp_path / "out")
    with pytest.raises(ValueError):
        manager.write_text(tmp_path / "evil.md", "x")
    with pytest.raises(ValueError):
        manager.write_json(tmp_path / "out" / ".." / "evil.json", {})


def test_json_serialisation_is_valid(tmp_path):
    manager = OutputManager(tmp_path)
    path = manager.write_json(tmp_path / "x.json", {"path": tmp_path, "items": (1, 2), "text": "naïve"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["items"] == [1, 2] and data["text"] == "naïve"
    assert not path.read_text(encoding="utf-8").startswith("```")


def _demo_state(topic="Edge Computing"):
    service = ResearchService([FixtureSearchProvider(topic)], delay_seconds=0, simulated=True)
    ctx = AgentContext(settings=load_settings(env={}), research_service=service, demo=True)
    return run_workflow(ctx, topic)


def test_save_run_writes_all_files(tmp_path):
    manager = OutputManager(tmp_path)
    paths = manager.save_run(_demo_state(), TODAY)
    assert set(paths.files) == {"blog", "research", "summary", "verification", "metadata"}
    for key in ("research", "summary", "verification", "metadata"):
        json.loads(paths.files[key].read_text(encoding="utf-8"))
    research = json.loads(paths.files["research"].read_text(encoding="utf-8"))
    assert research["mode"] == "demonstration" and research["sources"][0]["simulated"] is True
    assert research["sources"][0]["publication_date"] is None
    assert paths.files["blog"].read_text(encoding="utf-8").startswith("# ")

    runs = manager.list_runs()
    assert len(runs) == 1 and manager.latest_run().directory == paths.directory
    assert manager.load_json(paths.directory, "verification")["claims_checked"] == 6


def test_failed_run_saves_metadata_only(tmp_path):
    manager = OutputManager(tmp_path)
    state = {"topic": "x topic", "execution_metadata": {"execution_status": "failed"}, "errors": ["boom"]}
    paths = manager.save_run(state, TODAY)
    assert set(paths.files) == {"metadata"}
