"""Settings loaded from environment variables / .env and validated at startup."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MIN_WORDS, MAX_WORDS = 300, 5000


class ConfigError(Exception):
    """Raised when configuration values are invalid."""


class Settings(BaseModel):
    openrouter_api_key: SecretStr | None = None
    openrouter_model: str | None = None
    tavily_api_key: SecretStr | None = None
    blog_output_dir: Path = Field(Path("output"), validate_default=True)
    max_search_results: int = Field(8, ge=1, le=20)
    max_research_queries: int = Field(3, ge=1, le=5)
    blog_word_count: int = Field(1200, ge=MIN_WORDS, le=MAX_WORDS)
    verification_retries: int = Field(1, ge=0, le=3)
    request_timeout: int = Field(30, ge=5, le=120)
    log_level: str = "INFO"

    @field_validator("openrouter_api_key", "tavily_api_key", mode="before")
    @classmethod
    def _blank_secret_is_none(cls, value):
        return value.strip() if isinstance(value, str) and value.strip() else None

    @field_validator("openrouter_model", mode="before")
    @classmethod
    def _blank_model_is_none(cls, value):
        return value.strip() if isinstance(value, str) and value.strip() else None

    @field_validator("log_level", mode="before")
    @classmethod
    def _valid_log_level(cls, value):
        value = (value or "INFO").strip().upper()
        if value not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL")
        return value

    @field_validator("blog_output_dir", mode="before")
    @classmethod
    def _safe_output_dir(cls, value):
        """Restrict writes to a folder inside the project directory."""
        raw = Path(str(value or "output").strip() or "output")
        resolved = (raw if raw.is_absolute() else PROJECT_ROOT / raw).resolve()
        if not resolved.is_relative_to(PROJECT_ROOT):
            raise ValueError(f"must be inside the project folder ({PROJECT_ROOT})")
        return resolved

    # ------------------------------------------------------------------ helpers
    @property
    def llm_configured(self) -> bool:
        return self.openrouter_api_key is not None and self.openrouter_model is not None

    @property
    def tavily_configured(self) -> bool:
        return self.tavily_api_key is not None

    def missing_for_live_mode(self) -> list[str]:
        missing = []
        if self.openrouter_api_key is None:
            missing.append("OPENROUTER_API_KEY")
        if self.openrouter_model is None:
            missing.append("OPENROUTER_MODEL")
        return missing

    def secret_values(self) -> list[str]:
        """Secret strings that must never appear in logs."""
        return [s.get_secret_value() for s in (self.openrouter_api_key, self.tavily_api_key) if s]


_ENV_FIELDS = {
    "OPENROUTER_API_KEY": "openrouter_api_key",
    "OPENROUTER_MODEL": "openrouter_model",
    "TAVILY_API_KEY": "tavily_api_key",
    "BLOG_OUTPUT_DIR": "blog_output_dir",
    "MAX_SEARCH_RESULTS": "max_search_results",
    "MAX_RESEARCH_QUERIES": "max_research_queries",
    "BLOG_WORD_COUNT": "blog_word_count",
    "VERIFICATION_RETRIES": "verification_retries",
    "REQUEST_TIMEOUT": "request_timeout",
    "LOG_LEVEL": "log_level",
}


def load_settings(env: Mapping[str, str] | None = None, dotenv_path: Path | None = None) -> Settings:
    """Build Settings from ``env`` (if given) or from the real environment plus ``.env``.

    Real environment variables take precedence over values in the .env file.
    """
    if env is None:
        path = dotenv_path or PROJECT_ROOT / ".env"
        file_values = {k: v for k, v in dotenv_values(path).items() if v is not None} if path.exists() else {}
        env = {**file_values, **os.environ}

    data = {field: env[name] for name, field in _ENV_FIELDS.items() if name in env and env[name] != ""}
    try:
        return Settings(**data)
    except ValidationError as exc:
        reverse = {v: k for k, v in _ENV_FIELDS.items()}
        problems = [
            f"{reverse.get(str(err['loc'][0]), err['loc'][0])}: {err['msg']}" for err in exc.errors()
        ]
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(problems)) from None
