import logging

import pytest

from app.config import PROJECT_ROOT, ConfigError, load_settings
from app.logging_config import RedactingFormatter


def test_defaults_when_environment_is_empty():
    settings = load_settings(env={})
    assert settings.max_search_results == 8
    assert settings.max_research_queries == 3
    assert settings.blog_word_count == 1200
    assert settings.verification_retries == 1
    assert settings.request_timeout == 30
    assert settings.log_level == "INFO"
    assert settings.blog_output_dir == (PROJECT_ROOT / "output").resolve()


def test_values_are_loaded_and_converted():
    settings = load_settings(env={"MAX_SEARCH_RESULTS": "5", "BLOG_WORD_COUNT": "900", "LOG_LEVEL": "debug",
                                  "OPENROUTER_MODEL": " some/model:free "})
    assert settings.max_search_results == 5
    assert settings.blog_word_count == 900
    assert settings.log_level == "DEBUG"
    assert settings.openrouter_model == "some/model:free"


def test_missing_api_keys_are_reported_not_faked():
    settings = load_settings(env={"OPENROUTER_API_KEY": "   ", "TAVILY_API_KEY": ""})
    assert settings.openrouter_api_key is None
    assert not settings.llm_configured
    assert not settings.tavily_configured
    assert settings.missing_for_live_mode() == ["OPENROUTER_API_KEY", "OPENROUTER_MODEL"]


def test_key_without_model_is_not_configured():
    settings = load_settings(env={"OPENROUTER_API_KEY": "sk-or-abc"})
    assert not settings.llm_configured
    assert settings.missing_for_live_mode() == ["OPENROUTER_MODEL"]


def test_secrets_are_hidden_in_repr():
    settings = load_settings(env={"OPENROUTER_API_KEY": "sk-or-super-secret-value"})
    assert "super-secret" not in repr(settings)
    assert settings.secret_values() == ["sk-or-super-secret-value"]


@pytest.mark.parametrize("name,value", [
    ("MAX_SEARCH_RESULTS", "abc"), ("MAX_SEARCH_RESULTS", "0"), ("BLOG_WORD_COUNT", "50"),
    ("VERIFICATION_RETRIES", "99"), ("REQUEST_TIMEOUT", "1"), ("LOG_LEVEL", "LOUD"),
])
def test_invalid_values_raise_config_error(name, value):
    with pytest.raises(ConfigError) as info:
        load_settings(env={name: value})
    assert name in str(info.value)


def test_output_dir_outside_project_is_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(env={"BLOG_OUTPUT_DIR": "../../somewhere-else"})


def test_dotenv_file_is_read(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("BLOG_WORD_COUNT=700\nOPENROUTER_MODEL=test/model\n", encoding="utf-8")
    monkeypatch.delenv("BLOG_WORD_COUNT", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    settings = load_settings(dotenv_path=env_file)
    assert settings.blog_word_count == 700
    assert settings.openrouter_model == "test/model"


def test_log_formatter_redacts_secrets():
    formatter = RedactingFormatter("%(message)s", secrets=["my-configured-secret"])
    record = logging.LogRecord("x", logging.INFO, __file__, 1,
                               "key=%s other=sk-or-v1-abcdef123456 Authorization: Bearer abcdefghijkl",
                               ("my-configured-secret",), None)
    text = formatter.format(record)
    assert "my-configured-secret" not in text
    assert "sk-or-v1-abcdef123456" not in text
    assert "abcdefghijkl" not in text
    assert "REDACTED" in text
