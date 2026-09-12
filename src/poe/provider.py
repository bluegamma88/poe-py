"""OpenRouter chat completions, with streamed text and fragmented tool calls."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from poe.config import Config
from poe.events import Emit, Event


class ProviderError(RuntimeError):
    pass


def reasoning_text(message: dict) -> str:
    """Recover streamed thinking from a saved assistant message."""
    parts = [
        detail.get("text") or detail.get("summary") or ""
        for detail in message.get("reasoning_details") or []
        if isinstance(detail, dict)
    ]
    return "".join(parts) or message.get("reasoning") or ""


async def sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Parse SSE framing, including comments and multiline data fields."""
    parts: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if parts:
                yield "\n".join(parts)
                parts.clear()
        elif line.startswith("data:"):
            value = line[5:]
            parts.append(value[1:] if value.startswith(" ") else value)
    if parts:
        yield "\n".join(parts)


class OpenRouter:
    def __init__(self, config: Config, *, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        self.transport = transport

    async def complete(self, messages: list[dict], tools: list[dict], emit: Emit) -> dict:
        config = self.config
        payload = {
            "model": config.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
        try:
            async with httpx.AsyncClient(
                transport=self.transport,
                timeout=httpx.Timeout(120, connect=20),
                headers={"Authorization": f"Bearer {config.api_key}", "X-Title": "Poe Python"},
            ) as client:
                for attempt in range(3):
                    async with client.stream(
                        "POST",
                        f"{config.base_url}/chat/completions",
                        json=payload,
                    ) as response:
                        if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                            delay = 2**attempt
                            retry_after = response.headers.get("retry-after", "")
                            if retry_after.isdigit():
                                delay = min(int(retry_after), 30)
                            await emit(Event("status", f"Provider busy; retrying in {delay}s…"))
                        else:
                            if response.is_error:
                                await response.aread()
                                try:
                                    error = response.json().get("error", {})
                                    detail = error.get("message", str(error))
                                except (ValueError, AttributeError):
                                    detail = response.text[:500]
                                raise ProviderError(
                                    f"OpenRouter HTTP {response.status_code}: {detail}"
                                )
                            return await self._consume(response, emit)
                    # Retry only rejected requests; never replay a partially streamed round.
                    await asyncio.sleep(delay)
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenRouter connection failed: {exc}") from exc
        raise ProviderError("OpenRouter retry limit reached")

    async def _consume(self, response: httpx.Response, emit: Emit) -> dict:
        content: list[str] = []
        reasoning: list[str] = []
        details: dict[int, dict] = {}
        calls: dict[int, dict] = {}
        finish: str | None = None
        try:
            async for data in sse_data(response):
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    error = chunk["error"]
                    raise ProviderError(f"OpenRouter: {error.get('message', error)}")
                if chunk.get("usage"):
                    await emit(Event("usage", data=chunk["usage"]))
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                if text := delta.get("content"):
                    content.append(text)
                    await emit(Event("text", text))
                if thought := delta.get("reasoning") or delta.get("reasoning_content"):
                    reasoning.append(thought)
                    await emit(Event("reasoning", thought))
                for position, detail in enumerate(delta.get("reasoning_details") or []):
                    index = detail.get("index", position)
                    target = details.setdefault(index, {})
                    for key, value in detail.items():
                        if key in {"text", "summary", "data", "signature"} and isinstance(
                            value, str
                        ):
                            target[key] = target.get(key, "") + value
                            # Providers that send only reasoning_details stream the
                            # thinking here; do not repeat what `reasoning` carried.
                            if not thought and key in {"text", "summary"}:
                                await emit(Event("reasoning", value))
                        else:
                            target[key] = value
                for fragment in delta.get("tool_calls") or []:
                    index = fragment["index"]
                    call = calls.setdefault(
                        index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if fragment.get("id"):
                        call["id"] = fragment["id"]
                    function = fragment.get("function") or {}
                    for key in ("name", "arguments"):
                        if function.get(key):
                            call["function"][key] += function[key]
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ProviderError(f"Invalid response stream: {exc}") from exc

        if finish not in {"stop", "tool_calls"}:
            reason = finish or "connection ended without a finish reason"
            raise ProviderError(
                f"Incomplete response ({reason}); no tool calls from this round ran"
            )
        message: dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
        if details:
            message["reasoning_details"] = list(details.values())
        elif reasoning:
            message["reasoning"] = "".join(reasoning)
        if calls:
            ordered = [calls[index] for index in sorted(calls)]
            ids = [call["id"] for call in ordered]
            if len(set(ids)) != len(ids) or any(
                not call["id"] or not call["function"]["name"] for call in ordered
            ):
                raise ProviderError("Provider returned incomplete or duplicate tool calls")
            message["tool_calls"] = ordered
        elif finish == "tool_calls":
            raise ProviderError("Provider ended with tool_calls but supplied no calls")
        elif not content:
            raise ProviderError("The model returned no answer; try another prompt or model")
        return message
