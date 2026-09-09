"""Offline Stage 1 baseline for current LLM accounting and retry visibility.

This script exercises the existing compatibility and application seams with
scripted responses, plus a real pinned OpenAI client backed by httpx.MockTransport.
It never contacts a live provider. Run from MetaRec-backend.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from importlib.metadata import distributions, version
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Awaitable
from urllib.parse import urlparse

import httpx
from openai import OpenAI

# Keep invocation reproducible from MetaRec-backend without requiring an
# editable install of the application package.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from llm_compat import create_chat_completion, create_chat_completion_async
from llm_service import analyze_user_message
from llm_usage import UsageLedger, push_ledger, record_response_usage, reset_ledger


PRICE_TABLE = {"baseline-model": {"input": 2.0, "output": 8.0}}
VALID_INTENT = json.dumps(
    {"intent": "chat", "reply": "ok", "confidence": 0.9},
    separators=(",", ":"),
)


class ScriptedResponse:
    def __init__(self, content: str, usage: tuple[int, int] | None = (100, 20)):
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content=content), text=content)
        ]
        self.model = "baseline-model"
        self.usage = (
            SimpleNamespace(
                prompt_tokens=usage[0],
                completion_tokens=usage[1],
                total_tokens=usage[0] + usage[1],
            )
            if usage is not None
            else None
        )


class ScriptedCompletions:
    def __init__(self, outputs: list[Any]):
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> ScriptedResponse:
        self.calls.append(kwargs)
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple):
            return ScriptedResponse(item[0], item[1])
        return ScriptedResponse(str(item))


class ScriptedAsyncClient:
    def __init__(self, outputs: list[Any]):
        completions = ScriptedCompletions(outputs)
        self.chat = SimpleNamespace(completions=completions)


def _event(event: Any) -> dict[str, Any]:
    return {
        "model": event.model,
        "prompt_tokens": event.prompt_tokens,
        "completion_tokens": event.completion_tokens,
        "total_tokens": event.total_tokens,
        "cost_usd": event.cost_usd,
    }


async def _scoped(
    name: str,
    operation: Callable[[ScriptedAsyncClient], Awaitable[Any]],
    client: ScriptedAsyncClient,
) -> dict[str, Any]:
    ledger = UsageLedger()
    token = push_ledger(ledger)
    error: str | None = None
    result: Any = None
    try:
        result = await operation(client)
    except Exception as exc:  # baseline intentionally records failure behavior
        error = f"{type(exc).__name__}: {exc}"
    finally:
        reset_ledger(token)

    physical = len(client.chat.completions.calls)
    payload: dict[str, Any] = {
        "scenario": name,
        "logical_request_count": 1,
        "physical_provider_invocation_count": physical,
        "visible_usage_event_count": len(ledger.events),
        "invisible_attempt_count": max(0, physical - len(ledger.events)),
        "retry_count": max(0, physical - 1),
        "usage_events": [_event(item) for item in ledger.events],
        "totals": ledger.totals,
        "error": error,
    }
    if result is not None:
        payload["result_type"] = type(result).__name__
        if hasattr(result, "intent"):
            payload["intent"] = result.intent
            payload["confidence"] = result.confidence
    return payload


async def _response_with_usage(client: ScriptedAsyncClient) -> Any:
    response = await create_chat_completion_async(
        client,
        model="baseline-model",
        messages=[{"role": "user", "content": "test"}],
    )
    record_response_usage(response, "baseline-model")
    return response


async def _compatibility_retry(client: ScriptedAsyncClient) -> Any:
    response = await create_chat_completion_async(
        client,
        model="baseline-model",
        messages=[{"role": "user", "content": "test"}],
        temperature=0.2,
    )
    record_response_usage(response, "baseline-model")
    return response


async def _provider_exception(client: ScriptedAsyncClient) -> Any:
    return await _response_with_usage(client)


async def _analyze(
    client: ScriptedAsyncClient,
    *,
    max_retries: int,
) -> Any:
    return await analyze_user_message(
        client,
        "hello",
        model="baseline-model",
        max_format_retries=max_retries,
    )


def _sdk_retry_scenario() -> dict[str, Any]:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append({"method": request.method, "url": str(request.url)})
        if len(requests) == 1:
            return httpx.Response(
                500,
                json={"error": {"message": "synthetic server failure", "type": "server_error"}},
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-stage1",
                "object": "chat.completion",
                "created": 0,
                "model": "baseline-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )

    ledger = UsageLedger()
    token = push_ledger(ledger)
    error: str | None = None
    try:
        client = OpenAI(
            api_key="stage1-local-only",
            base_url="http://stage1.mock/v1",
            max_retries=2,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        try:
            response = create_chat_completion(
                client,
                model="baseline-model",
                messages=[{"role": "user", "content": "test"}],
            )
            record_response_usage(response, "baseline-model")
        finally:
            client.close()
    except Exception as exc:  # pragma: no cover - retained in raw baseline output
        error = f"{type(exc).__name__}: {exc}"
    finally:
        reset_ledger(token)

    return {
        "scenario": "sdk_automatic_retry_500_then_success",
        "logical_request_count": 1,
        "physical_provider_invocation_count": len(requests),
        "visible_usage_event_count": len(ledger.events),
        "invisible_attempt_count": max(0, len(requests) - len(ledger.events)),
        "retry_count": max(0, len(requests) - 1),
        "usage_events": [_event(item) for item in ledger.events],
        "totals": ledger.totals,
        "requests": requests,
        "error": error,
        "sdk_max_retries": 2,
    }


def _metadata() -> dict[str, Any]:
    safe_env_names = (
        "LLM_MODEL",
        "LLM_TIMEOUT_SECONDS",
        "LLM_SDK_MAX_RETRIES",
        "LLM_TRUST_ENV",
        "LLM_MAX_FORMAT_RETRIES",
        "LLM_TEXT_MAX_TOKENS",
        "LLM_PRICE_INPUT_PER_1M",
        "LLM_PRICE_OUTPUT_PER_1M",
        "LLM_PRICE_TABLE_JSON",
        "AGENT_PLANNING_MODEL",
        "AGENT_SUMMARY_MODEL",
        "AGENT_LLM_TIMEOUT_SECONDS",
        "ITINERARY_MAX_PROVIDER_CALLS",
        "ITINERARY_MAX_RETRIEVAL_ROUNDS",
        "ITINERARY_SOLVER",
        "METAREC_CHECKPOINTER_BACKEND",
    )
    try:
        git_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
        ).strip()
        git_status = subprocess.check_output(
            ["git", "status", "--short"], text=True, stderr=subprocess.STDOUT
        ).splitlines()
    except Exception as exc:
        git_revision = None
        git_status = [f"metadata error: {type(exc).__name__}: {exc}"]

    package_names = (
        "openai",
        "httpx",
        "fastapi",
        "pydantic",
        "langgraph",
        "langgraph-checkpoint-postgres",
        "pytest",
        "pytest-asyncio",
    )
    packages: dict[str, str] = {}
    for package in package_names:
        try:
            packages[package] = version(package)
        except Exception:
            packages[package] = "unavailable"

    all_packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in distributions()
        if distribution.metadata.get("Name")
    }

    source_hashes: dict[str, str] = {}
    for relative_path in (
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "pytest.ini",
        "client.py",
        "llm_compat.py",
        "llm_usage.py",
        "llm_service.py",
    ):
        path = BACKEND_ROOT / relative_path
        source_hashes[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()

    configured_env = {
        name: os.getenv(name)
        for name in safe_env_names
        if os.getenv(name) is not None
    }
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_COMPAT_BASE_URL")
    if base_url:
        configured_env["LLM_BASE_URL_HOST"] = urlparse(base_url).hostname

    return {
        "python": sys.version,
        "packages": packages,
        "all_installed_packages": dict(sorted(all_packages.items(), key=lambda item: item[0].lower())),
        "source_sha256": source_hashes,
        "git_revision": git_revision,
        "git_status_short": git_status,
        "safe_environment": configured_env,
        "synthetic_price_table": PRICE_TABLE,
        "script": Path(__file__).name,
        "command": " ".join(sys.argv),
        "live_provider_calls": False,
    }


async def _run() -> dict[str, Any]:
    # Make every cost result independent of the caller's live pricing settings.
    os.environ["LLM_PRICE_TABLE_JSON"] = json.dumps(PRICE_TABLE, separators=(",", ":"))
    scenarios = [
        await _scoped(
            "successful_response_with_usage",
            _response_with_usage,
            ScriptedAsyncClient([("ok", (100, 20))]),
        ),
        await _scoped(
            "successful_response_without_usage",
            _response_with_usage,
            ScriptedAsyncClient([("ok", None)]),
        ),
        await _scoped(
            "provider_exception",
            _provider_exception,
            ScriptedAsyncClient([RuntimeError("synthetic provider failure")]),
        ),
        await _scoped(
            "compatibility_retry",
            _compatibility_retry,
            ScriptedAsyncClient([
                RuntimeError("Unsupported value: temperature is not supported"),
                ("ok", (100, 20)),
            ]),
        ),
        await _scoped(
            "response_format_fallback",
            lambda client: _analyze(client, max_retries=0),
            ScriptedAsyncClient([
                RuntimeError("response_format is not supported"),
                (VALID_INTENT, (100, 20)),
            ]),
        ),
        await _scoped(
            "malformed_structured_output_no_retry",
            lambda client: _analyze(client, max_retries=0),
            ScriptedAsyncClient([("not valid json", (100, 20))]),
        ),
        await _scoped(
            "empty_structured_output_then_retry",
            lambda client: _analyze(client, max_retries=1),
            ScriptedAsyncClient([
                ("", (100, 20)),
                (VALID_INTENT, (100, 20)),
            ]),
        ),
    ]
    scenarios.append(_sdk_retry_scenario())
    return {"metadata": _metadata(), "scenarios": scenarios}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = asyncio.run(_run())
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
