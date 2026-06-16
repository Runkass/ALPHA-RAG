"""LLM generation: YandexGPT, OpenRouter, or GigaChat."""

from __future__ import annotations

from typing import Any, Protocol

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import get_settings
from .prompts import build_messages


class LLMPaymentRequired(Exception):
    """API quota / billing exhausted (HTTP 402)."""


class LLMClient(Protocol):
    def close(self) -> None: ...


def llm_configured() -> bool:
    settings = get_settings()
    if settings.llm_provider == "yandex":
        return bool(settings.yandex_api_key and settings.yandex_folder_id)
    if settings.llm_provider == "openrouter":
        return bool(settings.openrouter_api_key and settings.openrouter_model)
    if settings.llm_provider == "ollama":
        return True
    return bool(settings.gigachat_credentials)


def _openai_client_kwargs(provider: str) -> dict[str, Any]:
    settings = get_settings()
    if provider == "yandex":
        return {
            "api_key": settings.yandex_api_key,
            "base_url": settings.yandex_base_url,
            "project": settings.yandex_folder_id,
        }
    if provider == "openrouter":
        headers: dict[str, str] = {}
        if settings.openrouter_site_url:
            headers["HTTP-Referer"] = settings.openrouter_site_url
        if settings.openrouter_app_name:
            headers["X-Title"] = settings.openrouter_app_name
        kwargs: dict[str, Any] = {
            "api_key": settings.openrouter_api_key,
            "base_url": settings.openrouter_base_url,
        }
        if headers:
            kwargs["default_headers"] = headers
        return kwargs
    if provider == "ollama":
        return {
            "api_key": settings.ollama_api_key,
            "base_url": settings.ollama_base_url,
        }
    raise ValueError(f"Unsupported OpenAI-compatible provider: {provider}")


def make_client() -> Any:
    settings = get_settings()
    if settings.llm_provider in {"yandex", "openrouter", "ollama"}:
        return OpenAI(
            **_openai_client_kwargs(settings.llm_provider),
        )
    from gigachat import GigaChat

    return GigaChat(
        credentials=settings.gigachat_credentials,
        verify_ssl_certs=settings.gigachat_verify_ssl,
    )


def close_client(client: Any) -> None:
    if client is None:
        return
    if hasattr(client, "close"):
        client.close()


def _chat_model(provider: str) -> str:
    settings = get_settings()
    if provider == "yandex":
        if settings.yandex_model.startswith("gpt://"):
            return settings.yandex_model
        return f"gpt://{settings.yandex_folder_id}/{settings.yandex_model}"
    if provider == "openrouter":
        return settings.openrouter_model
    if provider == "ollama":
        return settings.ollama_model
    raise ValueError(f"Unsupported OpenAI-compatible provider: {provider}")


def _generate_openai_compat(
    question: str,
    context: str,
    provider: str,
    client: OpenAI | None,
) -> str:
    settings = get_settings()
    messages = build_messages(question, context)
    own = client is None
    if own:
        client = make_client()
    if settings.yandex_request_delay_ms > 0:
        import time

        time.sleep(settings.yandex_request_delay_ms / 1000.0)
    try:
        response = client.chat.completions.create(
            model=_chat_model(provider),
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        if status is None and hasattr(exc, "response"):
            status = getattr(exc.response, "status_code", None)
        if status == 402:
            raise LLMPaymentRequired(str(exc)) from exc
        raise
    finally:
        if own:
            close_client(client)


def _generate_gigachat(question: str, context: str, client: Any) -> str:
    from gigachat.exceptions import ResponseError
    from gigachat.models import Chat, Messages, MessagesRole

    settings = get_settings()
    role_map = {
        "system": MessagesRole.SYSTEM,
        "user": MessagesRole.USER,
        "assistant": MessagesRole.ASSISTANT,
    }
    messages = build_messages(question, context)
    giga_messages = [
        Messages(role=role_map[m["role"]], content=m["content"]) for m in messages
    ]
    own = client is None
    if own:
        client = make_client()
    try:
        response = client.chat(
            Chat(
                model=settings.gigachat_model,
                messages=giga_messages,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
        )
        return (response.choices[0].message.content or "").strip()
    except ResponseError as exc:
        if getattr(exc, "status_code", None) == 402:
            raise LLMPaymentRequired(str(exc)) from exc
        raise
    finally:
        if own:
            close_client(client)


def _retry_openai():
    from openai import RateLimitError

    return retry(
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=2, min=10, max=120),
        retry=retry_if_exception_type(RateLimitError),
    )


@_retry_openai()
def generate_slot_answer(messages: list[dict[str, str]], client: Any | None = None) -> str:
    settings = get_settings()
    if settings.llm_provider not in {"yandex", "openrouter", "ollama"}:
        raise ValueError("Slot generation requires OpenAI-compatible provider")
    own = client is None
    if own:
        client = make_client()
    try:
        response = client.chat.completions.create(
            model=_chat_model(settings.llm_provider),
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=min(settings.llm_max_tokens, 180),
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        if status is None and hasattr(exc, "response"):
            status = getattr(exc.response, "status_code", None)
        if status == 402:
            raise LLMPaymentRequired(str(exc)) from exc
        raise
    finally:
        if own:
            close_client(client)


@_retry_openai()
def generate_answer(question: str, context: str, client: Any | None = None) -> str:
    settings = get_settings()
    if settings.llm_provider in {"yandex", "openrouter", "ollama"}:
        return _generate_openai_compat(question, context, settings.llm_provider, client)
    return _generate_gigachat(question, context, client)


def normalize_answer(answer: str) -> str:
    from .postprocess import postprocess_answer

    return postprocess_answer(answer)


def make_async_client():
    from openai import AsyncOpenAI

    s = get_settings()
    if s.llm_provider not in {"yandex", "openrouter", "ollama"}:
        raise ValueError("Async generation is supported only for OpenAI-compatible providers")
    return AsyncOpenAI(**_openai_client_kwargs(s.llm_provider))


async def generate_answer_async(question: str, context: str, client) -> str:
    import asyncio
    from openai import RateLimitError
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=1, max=45),
        retry=retry_if_exception_type(RateLimitError),
    )
    async def _call():
        s = get_settings()
        resp = await client.chat.completions.create(
            model=_chat_model(s.llm_provider),
            messages=build_messages(question, context),
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    try:
        return await _call()
    except Exception as exc:
        status = getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        if status == 402:
            raise LLMPaymentRequired(str(exc)) from exc
        if status == 429:
            raise RateLimitError("rate limit", response=None, body=None) from exc
        raise


async def generate_batch_async(
    items: list[tuple[int, str, str]],
    client=None,
    *,
    concurrency: int | None = None,
) -> list[tuple[int, str]]:
    import asyncio

    s = get_settings()
    sem = asyncio.Semaphore(concurrency or s.llm_concurrency)
    own = client is None
    if own:
        client = make_async_client()
    out: list[tuple[int, str]] = []

    async def _one(q_id: int, query: str, context: str) -> None:
        async with sem:
            if s.yandex_request_delay_ms > 0:
                await asyncio.sleep(s.yandex_request_delay_ms / 1000.0)
            try:
                text = await generate_answer_async(query, context, client)
                out.append((q_id, text))
            except LLMPaymentRequired:
                raise
            except Exception:
                out.append((q_id, "Нет ответа"))

    await asyncio.gather(*[_one(a, b, c) for a, b, c in items])
    if own:
        await client.close()
    return out
