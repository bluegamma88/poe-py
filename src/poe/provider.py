"""OpenRouter chat completions, with streamed text and fragmented tool calls."""

from __future__ import annotations

import asyncio
import functools
import importlib.metadata
import json
import os
import platform
import random
import ssl
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from poe.config import Config
from poe.events import Emit, Event


class ProviderError(RuntimeError):
    pass


MAX_ATTEMPTS = 3
MAX_RETRY_DELAY = 30.0
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504, 524, 529}
RETRYABLE_TRANSPORT_ERRORS = (
    ssl.SSLError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.WriteError,
    httpx.WriteTimeout,
    httpx.RemoteProtocolError,
)
NETWORK_DEBUG_ENV = "POE_NETWORK_DEBUG"
NETWORK_DEBUG_PATH_ENV = "POE_NETWORK_DEBUG_PATH"
NETWORK_DEBUG_VALUES = {"1", "true", "yes", "on"}
SAFE_RESPONSE_HEADERS = ("cf-ray", "x-generation-id", "x-request-id", "date", "server")
TRACE_OPERATIONS = {
    "connection.connect_tcp",
    "connection.start_tls",
    "http11.send_request_headers",
    "http11.send_request_body",
    "http11.receive_response_headers",
    "http2.send_request_headers",
    "http2.send_request_body",
    "http2.receive_response_headers",
}


@dataclass
class _StreamState:
    started: bool = False
    events: int = 0
    data_bytes: int = 0
    first_event_seconds: float | None = None
    attempt_started: float = 0.0
    generation_id: str | None = None


def _network_debug_path() -> Path | None:
    if os.environ.get(NETWORK_DEBUG_ENV, "").strip().lower() not in NETWORK_DEBUG_VALUES:
        return None
    configured = os.environ.get(NETWORK_DEBUG_PATH_ENV, "").strip()
    return (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".poe" / "network-debug.jsonl"
    )


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_metadata() -> dict[str, str | None]:
    return {
        "python": sys.version.split()[0],
        "openssl": ssl.OPENSSL_VERSION,
        "httpx": httpx.__version__,
        "httpcore": _package_version("httpcore"),
        "anyio": _package_version("anyio"),
        "platform": platform.platform(),
    }


def _exception_chain(exc: BaseException) -> list[dict[str, str]]:
    result = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        result.append({"type": type(current).__name__, "message": str(current)[:2000]})
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return result


def _network_metadata(network: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if network is None or not hasattr(network, "get_extra_info"):
        return result
    try:
        result["local_address"] = network.get_extra_info("client_addr")
        result["remote_address"] = network.get_extra_info("server_addr")
        ssl_object = network.get_extra_info("ssl_object")
        if ssl_object is not None:
            result["tls_version"] = ssl_object.version()
            cipher = ssl_object.cipher()
            result["tls_cipher"] = cipher[0] if cipher else None
            result["alpn_protocol"] = ssl_object.selected_alpn_protocol()
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return result


def _response_metadata(response: httpx.Response) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status_code": response.status_code,
        "http_version": response.http_version,
        "headers": {
            name: response.headers[name]
            for name in SAFE_RESPONSE_HEADERS
            if name in response.headers
        },
    }
    result.update(_network_metadata(response.extensions.get("network_stream")))
    return result


def _stream_metadata(state: _StreamState) -> dict[str, int | float | str | None]:
    return {
        "sse_events": state.events,
        "sse_data_bytes": state.data_bytes,
        "first_sse_seconds": state.first_event_seconds,
        "generation_id": state.generation_id,
    }


class _NetworkDiagnostics:
    def __init__(self, model: str):
        self.path = _network_debug_path()
        self.model = model
        self.request_id = uuid4().hex
        self.started = time.monotonic()
        self._last_trace: dict[int, str] = {}

    async def trace(self, attempt: int, name: str, info: dict[str, Any]) -> None:
        operation, _, state = name.rpartition(".")
        if operation not in TRACE_OPERATIONS and not (
            operation in {"http11.receive_response_body", "http2.receive_response_body"}
            and state == "failed"
        ):
            return
        self._last_trace[attempt] = name
        data: dict[str, Any] = {"attempt": attempt, "operation": name}
        if state == "complete" and operation in {
            "connection.connect_tcp",
            "connection.start_tls",
        }:
            data["network"] = _network_metadata(info.get("return_value"))
        if state == "failed" and isinstance(info.get("exception"), BaseException):
            data["exceptions"] = _exception_chain(info["exception"])
        self.write("transport_trace", **data)

    def last_trace(self, attempt: int) -> str | None:
        return self._last_trace.get(attempt)

    def write(self, event: str, **data: Any) -> bool:
        if self.path is None:
            return False
        record = {
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "event": event,
            "request_id": self.request_id,
            "model": self.model,
            "elapsed_seconds": round(time.monotonic() - self.started, 6),
            **data,
        }
        descriptor: int | None = None
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            os.fchmod(descriptor, 0o600)
            target = os.fdopen(descriptor, "a", encoding="utf-8")
            descriptor = None
            with target:
                target.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
            return True
        except OSError:
            return False
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _retry_delay(attempt: int, retry_after: str = "") -> float:
    """Return a bounded Retry-After delay or exponential full jitter."""
    if retry_after.isdigit():
        return min(float(retry_after), MAX_RETRY_DELAY)
    if retry_after:
        try:
            retry_at = parsedate_to_datetime(retry_after)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            return min(max((retry_at - datetime.now(UTC)).total_seconds(), 0.0), MAX_RETRY_DELAY)
        except (TypeError, ValueError, OverflowError):
            pass
    return random.uniform(0, min(2**attempt, MAX_RETRY_DELAY))


def _delay_text(delay: float) -> str:
    return f"{delay:.1f}".rstrip("0").rstrip(".")


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
        self._debug_announced = False

    async def complete(self, messages: list[dict], tools: list[dict], emit: Emit) -> dict:
        config = self.config
        diagnostics = _NetworkDiagnostics(config.model)
        debugging = diagnostics.write("request_start", runtime=_runtime_metadata())
        if debugging and not self._debug_announced:
            await emit(Event("status", f"Network diagnostics: {diagnostics.path}"))
            self._debug_announced = True
        payload = {
            "model": config.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
        attempt_number = 0
        headers_received = False
        response_metadata: dict[str, Any] = {}
        state = _StreamState()
        unexpected_logged = False
        try:
            async with httpx.AsyncClient(
                transport=self.transport,
                timeout=httpx.Timeout(120, connect=20),
                headers={"Authorization": f"Bearer {config.api_key}", "X-Title": "Poe Python"},
            ) as client:
                for attempt in range(MAX_ATTEMPTS):
                    attempt_number = attempt + 1
                    headers_received = False
                    response_metadata = {}
                    state = _StreamState(attempt_started=time.monotonic())
                    retry_transport = True
                    diagnostics.write("attempt_start", attempt=attempt_number)
                    trace = functools.partial(diagnostics.trace, attempt_number)
                    try:
                        async with client.stream(
                            "POST",
                            f"{config.base_url}/chat/completions",
                            json=payload,
                            extensions={"trace": trace},
                        ) as response:
                            headers_received = True
                            response_metadata = _response_metadata(response)
                            diagnostics.write(
                                "response_headers",
                                attempt=attempt_number,
                                response=response_metadata,
                            )
                            if (
                                response.status_code in RETRYABLE_STATUS_CODES
                                and attempt < MAX_ATTEMPTS - 1
                            ):
                                delay = _retry_delay(
                                    attempt, response.headers.get("retry-after", "")
                                )
                                diagnostics.write(
                                    "retry_scheduled",
                                    attempt=attempt_number,
                                    reason="http_status",
                                    delay_seconds=round(delay, 6),
                                    response=response_metadata,
                                )
                                await emit(
                                    Event(
                                        "status",
                                        f"Provider busy; retrying in {_delay_text(delay)}s "
                                        f"(attempt {attempt + 2}/{MAX_ATTEMPTS})…",
                                    )
                                )
                            else:
                                if response.is_error:
                                    retry_transport = False
                                    await response.aread()
                                    try:
                                        error = response.json().get("error", {})
                                        detail = error.get("message", str(error))
                                    except (ValueError, AttributeError):
                                        detail = response.text[:500]
                                    diagnostics.write(
                                        "http_error",
                                        attempt=attempt_number,
                                        response=response_metadata,
                                    )
                                    raise ProviderError(
                                        f"OpenRouter HTTP {response.status_code}: {detail}"
                                    )
                                try:
                                    message = await self._consume(response, emit, state)
                                except ProviderError as exc:
                                    diagnostics.write(
                                        "stream_error",
                                        attempt=attempt_number,
                                        phase="streaming" if state.started else "before_sse",
                                        error_type=type(exc).__name__,
                                        response=response_metadata,
                                        stream=_stream_metadata(state),
                                    )
                                    raise
                                diagnostics.write(
                                    "attempt_complete",
                                    attempt=attempt_number,
                                    response=response_metadata,
                                    stream=_stream_metadata(state),
                                )
                                return message
                    except RETRYABLE_TRANSPORT_ERRORS as exc:
                        phase = (
                            "streaming"
                            if state.started
                            else "before_sse"
                            if headers_received
                            else "before_headers"
                        )
                        attempts_remain = attempt < MAX_ATTEMPTS - 1
                        will_retry = retry_transport and not state.started and attempts_remain
                        diagnostics.write(
                            "transport_error",
                            attempt=attempt_number,
                            phase=phase,
                            will_retry=will_retry,
                            exceptions=_exception_chain(exc),
                            last_trace=diagnostics.last_trace(attempt_number),
                            response=response_metadata or None,
                            stream=_stream_metadata(state),
                        )
                        if not will_retry:
                            phase = " after response streaming began" if state.started else ""
                            attempts = (
                                f" after {MAX_ATTEMPTS} attempts"
                                if not state.started and attempt == MAX_ATTEMPTS - 1
                                else ""
                            )
                            raise ProviderError(
                                f"OpenRouter connection failed{phase}{attempts} "
                                f"({type(exc).__name__}): {exc}"
                            ) from exc
                        delay = _retry_delay(attempt)
                        diagnostics.write(
                            "retry_scheduled",
                            attempt=attempt_number,
                            reason="transport_error",
                            delay_seconds=round(delay, 6),
                        )
                        await emit(
                            Event(
                                "status",
                                f"Connection interrupted before response; retrying in "
                                f"{_delay_text(delay)}s "
                                f"(attempt {attempt + 2}/{MAX_ATTEMPTS})…",
                            )
                        )
                    except ProviderError:
                        raise
                    except httpx.HTTPError:
                        raise
                    except Exception as exc:
                        phase = (
                            "streaming"
                            if state.started
                            else "before_sse"
                            if headers_received
                            else "before_headers"
                        )
                        diagnostics.write(
                            "unexpected_error",
                            attempt=attempt_number,
                            phase=phase,
                            exceptions=_exception_chain(exc),
                            last_trace=diagnostics.last_trace(attempt_number),
                            response=response_metadata or None,
                            stream=_stream_metadata(state),
                        )
                        unexpected_logged = True
                        raise
                    # Retry only rejected requests or failures before the first SSE event;
                    # never replay a partially streamed round.
                    await asyncio.sleep(delay)
        except httpx.HTTPError as exc:
            diagnostics.write(
                "transport_error",
                attempt=attempt_number or None,
                phase=(
                    "streaming"
                    if state.started
                    else "before_sse"
                    if headers_received
                    else "client_setup"
                ),
                will_retry=False,
                exceptions=_exception_chain(exc),
                last_trace=diagnostics.last_trace(attempt_number),
                response=response_metadata or None,
                stream=_stream_metadata(state),
            )
            raise ProviderError(
                f"OpenRouter connection failed ({type(exc).__name__}): {exc}"
            ) from exc
        except ssl.SSLError as exc:
            diagnostics.write(
                "transport_error",
                attempt=attempt_number or None,
                phase="streaming" if state.started else "client_teardown",
                will_retry=False,
                exceptions=_exception_chain(exc),
                last_trace=diagnostics.last_trace(attempt_number),
                response=response_metadata or None,
                stream=_stream_metadata(state),
            )
            raise ProviderError(
                f"OpenRouter connection failed ({type(exc).__name__}): {exc}"
            ) from exc
        except ProviderError:
            raise
        except Exception as exc:
            if not unexpected_logged:
                diagnostics.write(
                    "unexpected_error",
                    attempt=attempt_number or None,
                    phase="client_setup" if not attempt_number else "client_teardown",
                    exceptions=_exception_chain(exc),
                    last_trace=diagnostics.last_trace(attempt_number),
                    response=response_metadata or None,
                    stream=_stream_metadata(state),
                )
            raise
        raise ProviderError("OpenRouter retry limit reached")

    async def _consume(
        self, response: httpx.Response, emit: Emit, state: _StreamState | None = None
    ) -> dict:
        content: list[str] = []
        reasoning: list[str] = []
        details: dict[int, dict] = {}
        calls: dict[int, dict] = {}
        finish: str | None = None
        try:
            async for data in sse_data(response):
                if state is not None:
                    state.events += 1
                    state.data_bytes += len(data.encode())
                    if not state.started:
                        state.first_event_seconds = round(
                            time.monotonic() - state.attempt_started, 6
                        )
                    state.started = True
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if (
                    state is not None
                    and state.generation_id is None
                    and isinstance(chunk.get("id"), str)
                ):
                    state.generation_id = chunk["id"][:200]
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
