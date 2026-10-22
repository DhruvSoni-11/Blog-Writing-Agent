"""Shared LLM interface. Real calls go to OpenRouter through its OpenAI-compatible API."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol, TypeVar

import httpx
import openai
from pydantic import BaseModel, ValidationError

from app.config import OPENROUTER_BASE_URL, Settings
from app.utils.text_utils import extract_json

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

MAX_BACKOFF_SECONDS = 20.0
MAX_TOKEN_CEILING = 16000  # upper bound when a truncated reply is retried with a larger budget


# --------------------------------------------------------------------------- errors
class LLMError(Exception):
    """Base class for LLM failures. Messages are safe to show to the user (no secrets)."""


class LLMAuthError(LLMError):
    pass


class LLMModelUnavailableError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMQuotaError(LLMError):
    pass


class LLMResponseError(LLMError):
    pass


class LLMConnectionError(LLMError):
    pass


class LLMTruncatedError(LLMResponseError):
    """The reply hit max_tokens. Reasoning models spend part of the budget on hidden thinking."""


# --------------------------------------------------------------------------- interface
class LLMClient(Protocol):
    model_name: str

    def complete(self, system: str, user: str, *, purpose: str, max_tokens: int = 2048,
                 temperature: float = 0.2) -> str: ...


class OpenRouterClient:
    """Calls one configured OpenRouter model with timeouts and a bounded retry policy."""

    def __init__(self, api_key: str, model: str, timeout: float = 30, max_attempts: int = 3,
                 sdk_client: openai.OpenAI | None = None, sleep=time.sleep) -> None:
        if not api_key or not model:
            raise LLMError("OPENROUTER_API_KEY and OPENROUTER_MODEL must both be set.")
        self.model_name = model
        self.max_attempts = max(1, min(max_attempts, 5))
        self._sleep = sleep
        # SDK-level retries are disabled; this class owns the (bounded) retry loop.
        self._client = sdk_client or openai.OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            timeout=timeout,
            max_retries=0,
            default_headers={"X-Title": "BlogForge AI"},
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenRouterClient:
        if not settings.llm_configured:
            raise LLMError(
                "OpenRouter is not configured. Set OPENROUTER_API_KEY and OPENROUTER_MODEL in .env."
            )
        return cls(settings.openrouter_api_key.get_secret_value(), settings.openrouter_model,
                   timeout=settings.request_timeout)

    def complete(self, system: str, user: str, *, purpose: str, max_tokens: int = 2048,
                 temperature: float = 0.2) -> str:
        last_error: LLMError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                logger.info("LLM call purpose=%s model=%s attempt=%d", purpose, self.model_name, attempt)
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return self._extract_text(response)
            except openai.AuthenticationError:
                raise LLMAuthError("OpenRouter rejected the API key (401). Check OPENROUTER_API_KEY.") from None
            except openai.PermissionDeniedError as exc:
                raise LLMModelUnavailableError(
                    f"OpenRouter denied access to model '{self.model_name}' (403): {_short(exc)}"
                ) from None
            except openai.NotFoundError:
                raise LLMModelUnavailableError(self._unavailable_message()) from None
            except openai.RateLimitError as exc:
                last_error = LLMRateLimitError(
                    "OpenRouter rate limit or free-tier quota reached (429). Wait and retry, "
                    "or choose another free model. Details: " + _short(exc)
                )
                delay = _retry_after(exc, default=2.0 * attempt)
            except openai.APIStatusError as exc:
                if exc.status_code == 402:
                    raise LLMQuotaError(
                        "OpenRouter reports insufficient credits (402). Free-tier limits may be exhausted "
                        "or the configured model is not free."
                    ) from None
                if exc.status_code == 400 and "model" in _short(exc).lower():
                    raise LLMModelUnavailableError(self._unavailable_message() + f" ({_short(exc)})") from None
                if exc.status_code < 500 and exc.status_code != 408:
                    raise LLMError(f"OpenRouter request failed ({exc.status_code}): {_short(exc)}") from None
                last_error = LLMConnectionError(f"OpenRouter server error ({exc.status_code}).")
                delay = 2.0 * attempt
            except (openai.APITimeoutError, openai.APIConnectionError):
                last_error = LLMConnectionError("Could not reach OpenRouter (network error or timeout).")
                delay = 2.0 * attempt
            except LLMTruncatedError as exc:
                last_error = exc
                delay = 0.0
                max_tokens = min(max_tokens * 2, MAX_TOKEN_CEILING)  # bounded: doubles at most a few times
            except LLMResponseError as exc:
                last_error = exc
                delay = 1.0

            logger.warning("LLM call %s failed (attempt %d/%d): %s", purpose, attempt, self.max_attempts, last_error)
            if attempt < self.max_attempts and delay:
                self._sleep(min(delay, MAX_BACKOFF_SECONDS))
        raise last_error or LLMError("LLM call failed")

    def _extract_text(self, response) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            # OpenRouter can return HTTP 200 with an error object instead of choices.
            error = getattr(response, "error", None) or (getattr(response, "model_extra", None) or {}).get("error")
            raise LLMResponseError(f"OpenRouter returned no choices{f': {error}' if error else ''}.")
        content = getattr(choices[0].message, "content", None)
        if getattr(choices[0], "finish_reason", None) == "length":
            raise LLMTruncatedError("The model's reply was cut off at the token limit.")
        if not content or not content.strip():
            raise LLMResponseError("OpenRouter returned an empty message.")
        return content.strip()

    def _unavailable_message(self) -> str:
        return (
            f"Model '{self.model_name}' is not available on OpenRouter. Pick a currently available "
            "free model from https://openrouter.ai/models (filter by price: free) and set OPENROUTER_MODEL."
        )


def _short(exc: Exception, limit: int = 300) -> str:
    message = getattr(exc, "message", None) or str(exc)
    return message[:limit]


def _retry_after(exc: openai.APIStatusError, default: float) -> float:
    try:
        return float(exc.response.headers.get("retry-after", default))
    except (AttributeError, TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- structured output
def complete_json(llm: LLMClient, system: str, user: str, model_cls: type[T], *, purpose: str,
                  max_tokens: int = 2048, context: dict | None = None, prepare=None) -> T:
    """Ask for JSON, validate it with ``model_cls`` and make exactly one repair attempt.

    ``prepare`` may clean the parsed data (e.g. drop unknown source IDs) before validation.
    """
    reply = llm.complete(system, user, purpose=purpose, max_tokens=max_tokens)
    for attempt in (1, 2):
        try:
            data = extract_json(reply)
            if prepare is not None:
                data = prepare(data)
            return model_cls.model_validate(data, context=context)
        except (ValueError, ValidationError, TypeError, AttributeError) as exc:
            problem = str(exc)[:800]
            logger.warning("Invalid structured output for %s (attempt %d): %s", purpose, attempt, problem)
            if attempt == 2:
                raise LLMResponseError(f"The model returned invalid structured data for {purpose}.") from None
            repair = (
                f"{user}\n\nYour previous reply could not be used because: {problem}\n"
                "Return ONLY a single valid JSON object that follows the requested schema. "
                "No markdown fences, no commentary."
            )
            reply = llm.complete(system, repair, purpose=f"{purpose}.repair", max_tokens=max_tokens)
    raise LLMResponseError(f"Structured output failed for {purpose}.")  # pragma: no cover


# --------------------------------------------------------------------------- model check
@dataclass
class ModelCheck:
    reachable: bool
    available: bool
    is_free: bool
    message: str


def check_model_availability(model: str, timeout: float = 15, http_get=None) -> ModelCheck:
    """Look the model up in OpenRouter's public model list and confirm its price is zero.

    The model list endpoint needs no API key, so no secret is sent.
    """
    get = http_get or httpx.get
    try:
        response = get(f"{OPENROUTER_BASE_URL}/models", timeout=timeout)
        response.raise_for_status()
        models = response.json().get("data", [])
    except Exception as exc:  # network, HTTP or JSON problems
        logger.warning("Model availability check failed: %s", type(exc).__name__)
        return ModelCheck(False, False, False, "Could not reach OpenRouter to check the model list.")

    entry = next((m for m in models if m.get("id") == model), None)
    if entry is None:
        return ModelCheck(True, False, False,
                          f"Model '{model}' is not in OpenRouter's current model list. Choose an available "
                          "free model at https://openrouter.ai/models.")
    pricing = entry.get("pricing") or {}
    try:
        is_free = all(float(pricing.get(k, "1") or 0) == 0 for k in ("prompt", "completion"))
    except (TypeError, ValueError):
        is_free = False
    if not is_free:
        return ModelCheck(True, True, False,
                          f"Model '{model}' is available but NOT free (prompt={pricing.get('prompt')}, "
                          f"completion={pricing.get('completion')}). BlogForge only runs with free models.")
    return ModelCheck(True, True, True, f"Model '{model}' is available and listed with $0 pricing.")
