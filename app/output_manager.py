"""Saves each run into its own folder: blog.md plus research/summary/verification/metadata JSON."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.utils.text_utils import slugify

logger = logging.getLogger(__name__)

FILES = {
    "blog": "blog.md",
    "research": "research.json",
    "summary": "summary.json",
    "verification": "verification.json",
    "metadata": "run_metadata.json",
}


@dataclass
class RunPaths:
    directory: Path
    files: dict[str, Path]


@dataclass
class RunInfo:
    directory: Path
    topic: str
    title: str | None
    status: str
    mode: str
    finished: str


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


class OutputManager:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir).resolve()

    # ------------------------------------------------------------------ paths
    def _inside_base(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.base_dir):
            raise ValueError(f"Refusing to write outside the output directory: {resolved}")
        return resolved

    def create_run_dir(self, topic: str, today: date | None = None) -> Path:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{(today or date.today()).isoformat()}-{slugify(topic)}"
        for n in range(1, 1000):
            candidate = self._inside_base(self.base_dir / (stem if n == 1 else f"{stem}-{n}"))
            try:
                candidate.mkdir()  # atomic: fails if it already exists, so runs never overwrite
                return candidate
            except FileExistsError:
                continue
        raise RuntimeError("Could not create a unique output directory")

    def write_json(self, path: Path, data: Any) -> Path:
        path = self._inside_base(path)
        text = json.dumps(to_jsonable(data), indent=2, ensure_ascii=False)
        json.loads(text)  # guarantees the file is valid JSON
        path.write_text(text + "\n", encoding="utf-8")
        return path

    def write_text(self, path: Path, text: str) -> Path:
        path = self._inside_base(path)
        path.write_text(text, encoding="utf-8")
        return path

    # ------------------------------------------------------------------ runs
    def save_run(self, state: dict, today: date | None = None) -> RunPaths:
        """Save whatever the run produced. Failed runs still get metadata for troubleshooting."""
        directory = self.create_run_dir(state.get("topic") or "untitled", today)
        files: dict[str, Path] = {}
        final = state.get("final_blog")
        if final is not None:
            files["blog"] = self.write_text(directory / FILES["blog"], final.markdown)
        for key, state_key in (("research", "research_package"), ("summary", "research_brief"),
                               ("verification", "verification_report")):
            if state.get(state_key) is not None:
                files[key] = self.write_json(directory / FILES[key], state[state_key])

        metadata = dict(state.get("execution_metadata") or {})
        metadata.update({
            "topic": state.get("topic"), "audience": state.get("audience"), "tone": state.get("tone"),
            "requested_word_count": state.get("requested_word_count"),
            "blog_title": final.title if final else None,
            "blog_word_count": final.word_count if final else None,
            "validation": state.get("validation"),
        })
        files["metadata"] = self.write_json(directory / FILES["metadata"], metadata)
        logger.info("Saved run to %s", directory)
        return RunPaths(directory, files)

    def list_runs(self) -> list[RunInfo]:
        if not self.base_dir.exists():
            return []
        runs = []
        for directory in self.base_dir.iterdir():
            meta_file = directory / FILES["metadata"]
            if not directory.is_dir() or not meta_file.exists():
                continue
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            runs.append(RunInfo(directory, meta.get("topic") or directory.name, meta.get("blog_title"),
                                meta.get("execution_status") or "unknown", meta.get("mode") or "unknown",
                                meta.get("workflow_end") or ""))
        return sorted(runs, key=lambda r: (r.finished, r.directory.name), reverse=True)

    def latest_run(self) -> RunInfo | None:
        runs = self.list_runs()
        return runs[0] if runs else None

    @staticmethod
    def load_json(directory: Path, key: str) -> dict | None:
        path = directory / FILES[key]
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
