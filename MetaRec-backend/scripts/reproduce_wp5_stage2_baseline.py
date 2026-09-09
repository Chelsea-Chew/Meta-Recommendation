"""Offline Phase 1 Stage 2 representative-workload baseline.

This runner exercises the current MetaRec service and graph seams with
deterministic scripted LLM responses and fake repositories/tools.  It never
contacts an LLM provider or an external API.  The experiment intentionally
keeps Stage 1's fault-injection scenarios separate: Stage 2 measures the
normal workload shape and accounting visibility of representative workflows.

Run from ``MetaRec-backend``::

    python scripts/reproduce_wp5_stage2_baseline.py

The output defaults to ``artifacts/stage2_representative_workload_raw.json``.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import business_repositories  # noqa: E402
import llm_usage  # noqa: E402
from llm_service import propose_gather_action  # noqa: E402
from langgraph_metarec.checkpointing import RuntimeCheckpointer  # noqa: E402
from langgraph_metarec.graphs.generic_graph import (  # noqa: E402
    GenericGraphAdapters,
    run_generic_domain_graph,
)
from langgraph_metarec.graphs.restaurant_graph import (  # noqa: E402
    RestaurantGraphAdapters,
    run_restaurant_graph,
)
from langgraph_metarec.tool_registry import ToolRegistry, ToolSpec  # noqa: E402
from service import MetaRecService  # noqa: E402


MODEL = "stage2-model"
PRICE_TABLE = {MODEL: {"input": 2.0, "output": 8.0}}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _intent_payload(*, intent: str, preferences: Optional[Dict[str, Any]] = None) -> str:
    return _json(
        {
            "intent": intent,
            "reply": "I can help with that.",
            "confidence": 0.9,
            "preferences": preferences,
        }
    )


@dataclass(frozen=True)
class ResponseSpec:
    content: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    tool_calls: Optional[List[Dict[str, Any]]] = None


class FakeResponse:
    def __init__(self, spec: ResponseSpec):
        message = SimpleNamespace(
            content=spec.content,
            tool_calls=spec.tool_calls or [],
        )
        self.choices = [SimpleNamespace(message=message, text=spec.content)]
        self.model = MODEL
        self.usage = SimpleNamespace(
            prompt_tokens=spec.prompt_tokens,
            completion_tokens=spec.completion_tokens,
            total_tokens=spec.prompt_tokens + spec.completion_tokens,
        )


class ScriptedCompletions:
    """A sync/async-compatible fake that records every physical invocation."""

    def __init__(
        self,
        *,
        async_responses: Dict[str, ResponseSpec],
        sync_responses: Dict[str, ResponseSpec],
        scope_by_label: Dict[str, str],
    ):
        self.async_responses = dict(async_responses)
        self.sync_responses = dict(sync_responses)
        self.scope_by_label = dict(scope_by_label)
        self.calls: List[Dict[str, Any]] = []

    @staticmethod
    def _label(kwargs: Dict[str, Any], *, sync: bool) -> str:
        messages = kwargs.get("messages") or []
        first = str(messages[0].get("content") or "") if messages else ""
        combined = "\n".join(str(item.get("content") or "") for item in messages)
        if sync:
            return "restaurant.plan" if kwargs.get("tools") else "restaurant.summary"
        if "Previous summary:" in combined:
            return "conversation.summary"
        if "Translate an itinerary request" in first:
            return "itinerary.constraints"
        if "You generate a recommendation confirmation" in combined:
            return "confirmation.payload"
        if "You refine a recommendation candidate search" in first:
            return "generic.gather_reasoner"
        return "intention.analyze"

    def _call(self, kwargs: Dict[str, Any], *, sync: bool) -> FakeResponse:
        label = self._label(kwargs, sync=sync)
        responses = self.sync_responses if sync else self.async_responses
        try:
            spec = responses[label]
        except KeyError as exc:
            raise RuntimeError(f"No scripted Stage 2 response for {label}") from exc
        self.calls.append(
            {
                "physical_attempt": len(self.calls) + 1,
                "call_site": label,
                "scope": self.scope_by_label.get(label, "unknown"),
                "channel": "sync" if sync else "async",
                "model": kwargs.get("model"),
                "temperature": kwargs.get("temperature"),
                "response_format": kwargs.get("response_format"),
                "tool_names": [
                    str(item.get("function", {}).get("name"))
                    for item in (kwargs.get("tools") or [])
                    if isinstance(item, dict)
                ],
                "message_roles": [item.get("role") for item in (kwargs.get("messages") or [])],
                "message_characters": sum(
                    len(str(item.get("content") or ""))
                    for item in (kwargs.get("messages") or [])
                ),
            }
        )
        return FakeResponse(spec)

    async def create(self, **kwargs: Any) -> FakeResponse:
        return self._call(kwargs, sync=False)

    def create_sync(self, **kwargs: Any) -> FakeResponse:
        return self._call(kwargs, sync=True)


class FakeClient:
    def __init__(
        self,
        *,
        channel: str = "async",
        async_responses: Optional[Dict[str, ResponseSpec]] = None,
        sync_responses: Optional[Dict[str, ResponseSpec]] = None,
        scope_by_label: Optional[Dict[str, str]] = None,
    ):
        completions = ScriptedCompletions(
            async_responses=async_responses or {},
            sync_responses=sync_responses or {},
            scope_by_label=scope_by_label or {},
        )
        self.completions = completions
        create = completions.create_sync if channel == "sync" else completions.create
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


class FakeUsageRepository:
    def __init__(self) -> None:
        self.flushes: List[Dict[str, Any]] = []

    async def record_events(self, *, user_id, conversation_id, task_id, events):
        self.flushes.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "task_id": task_id,
                "events": list(events),
            }
        )
        return len(events)


class FakeConversationRepository:
    def __init__(self, conversation: Dict[str, Any]) -> None:
        self.conversation = conversation
        self.summary_updates: List[Dict[str, Any]] = []

    async def get_full_conversation(self, user_id: str, conversation_id: str):
        return self.conversation

    async def update_conversation_context_summary(
        self, user_id: str, conversation_id: str, summary: str, watermark_id: str
    ):
        self.summary_updates.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "summary": summary,
                "watermark_id": watermark_id,
            }
        )
        return True


def _response(content: Optional[str], prompt: int, completion: int, *, tool_calls=None) -> ResponseSpec:
    return ResponseSpec(content, prompt, completion, tool_calls)


def _new_service(client: FakeClient) -> MetaRecService:
    service = MetaRecService(
        async_client=client,
        sync_client=client,
        summary_model=MODEL,
        planning_model=MODEL,
        llm_model=MODEL,
    )
    # These are the same in-memory seams used by the existing unit-test harness.
    service.profile_repository = None
    service.task_repository = None
    service.result_repository = None
    service.runtime_checkpointer = RuntimeCheckpointer(backend="memory")
    service.llm_max_format_retries = 0
    return service


def _summary_conversation() -> Dict[str, Any]:
    messages = []
    for index in range(8):
        role = "user" if index % 2 == 0 else "assistant"
        messages.append(
            {
                "id": f"m{index}",
                "role": role,
                "content": (
                    "I prefer affordable spicy food near Bugis."
                    if role == "user"
                    else "Assistant recommendation turn."
                ),
                "branch_id": "main",
                "metadata": {"message_id": f"m{index}"},
            }
        )
    return {
        "active_branch_id": "main",
        "preferences": {"location": "Bugis"},
        "metadata": {"context_summary": {"summary": "Previous summary."}},
        "messages": messages,
    }


def _fake_attraction_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="gmap.attraction.search",
            domain="attraction",
            tags={"#place", "#attraction"},
            input_schema={"type": "object"},
            output_schema={"type": "array"},
            adapter=lambda _params: [
                {
                    "title": "Fixture Museum",
                    "address": "Sentosa",
                    "rating": 4.6,
                    "reviews": 120,
                    "type": "Museum",
                }
            ],
        )
    )
    return registry


def _fake_restaurant_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="gmap.search",
            domain="restaurant",
            tags={"#place", "#restaurant"},
            input_schema={"type": "object"},
            output_schema={"type": "array"},
            adapter=lambda _params: [
                {
                    "title": "Fixture Bistro",
                    "address": "Bugis, Singapore",
                    "rating": 4.7,
                    "reviews": 220,
                    "type": "Casual Dining",
                }
            ],
        )
    )
    return registry


def _event_payload(event: Any) -> Dict[str, Any]:
    return {
        "model": event.model,
        "prompt_tokens": event.prompt_tokens,
        "completion_tokens": event.completion_tokens,
        "total_tokens": event.total_tokens,
        "cost_usd": event.cost_usd,
    }


def _aggregate_events(events: Iterable[Any]) -> Dict[str, Any]:
    events = list(events)
    return {
        "prompt_tokens": sum(event.prompt_tokens for event in events),
        "completion_tokens": sum(event.completion_tokens for event in events),
        "total_tokens": sum(event.total_tokens for event in events),
        "cost_usd": round(sum(event.cost_usd for event in events), 6),
    }


def _workload_record(
    *,
    workload_id: str,
    title: str,
    execution_level: str,
    entrypoint: str,
    fixture_id: str,
    client: FakeClient,
    events: List[Any],
    result_summary: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    calls = list(client.completions.calls)
    counts: Dict[str, int] = {}
    for call in calls:
        counts[call["call_site"]] = counts.get(call["call_site"], 0) + 1
    logical_requests = len(calls)
    retry_count = sum(max(0, count - 1) for count in counts.values())
    payload = {
        "workload_id": workload_id,
        "title": title,
        "execution_level": execution_level,
        "entrypoint": entrypoint,
        "fixture_id": fixture_id,
        "logical_llm_requests": logical_requests,
        "logical_requests_by_call_site": counts,
        "physical_attempts": len(calls),
        "physical_attempts_by_call_site": counts,
        "visible_usage_events": len(events),
        "invisible_attempts": max(0, len(calls) - len(events)),
        "retries_or_fallbacks": {
            "retry_count": retry_count,
            "fallbacks_observed": [],
            "note": "No fault was injected; Stage 1 contains mechanism-level fault scenarios.",
        },
        "usage": _aggregate_events(events),
        "usage_events": [_event_payload(event) for event in events],
        "llm_call_sites": calls,
        "result_summary": result_summary,
    }
    if extra:
        payload.update(extra)
    return payload


async def _normal_recommendation() -> Dict[str, Any]:
    client = FakeClient(
        async_responses={
            "intention.analyze": _response(
                _intent_payload(
                    intent="query",
                    preferences={
                        "restaurant_types": ["casual"],
                        "flavor_profiles": ["spicy"],
                        "dining_purpose": "friends",
                        "budget_range": {"min": 20, "max": 60, "currency": "SGD", "per": "person"},
                        "location": "Chinatown",
                    },
                ),
                120,
                30,
            ),
            "confirmation.payload": _response(
                _json({"message": "I will look for spicy casual restaurants in Chinatown. Is that correct?", "quick_actions": []}),
                180,
                35,
            ),
        },
        scope_by_label={"intention.analyze": "request", "confirmation.payload": "request"},
    )
    usage_repo = FakeUsageRepository()
    original_usage = business_repositories.usage_repository
    business_repositories.usage_repository = usage_repo
    try:
        service = _new_service(client)
        result = await service.handle_user_request_async(
            "Find spicy casual restaurants in Chinatown",
            user_id="stage2-normal-user",
            session_id=None,
            conversation_history=[],
        )
    finally:
        business_repositories.usage_repository = original_usage
    events = [event for flush in usage_repo.flushes for event in flush["events"]]
    return _workload_record(
        workload_id="normal_recommendation_request",
        title="Normal recommendation request",
        execution_level="service-level",
        entrypoint="MetaRecService.handle_user_request_async",
        fixture_id="normal-restaurant-confirmation-v1",
        client=client,
        events=events,
        result_summary={
            "type": result.get("type"),
            "domain": result.get("domain"),
            "routing_mode": (result.get("routing") or {}).get("mode"),
            "task_created": "task_id" in result,
        },
        extra={"usage_flush_count": len(usage_repo.flushes)},
    )


async def _background_summary() -> Dict[str, Any]:
    client = FakeClient(
        async_responses={
            "intention.analyze": _response(_intent_payload(intent="chat"), 120, 30),
            "conversation.summary": _response("User prefers affordable spicy food near Bugis.", 160, 40),
        },
        scope_by_label={"intention.analyze": "request", "conversation.summary": "background"},
    )
    usage_repo = FakeUsageRepository()
    conversation_repo = FakeConversationRepository(_summary_conversation())
    original_usage = business_repositories.usage_repository
    original_conversation = business_repositories.conversation_repository
    business_repositories.usage_repository = usage_repo
    business_repositories.conversation_repository = conversation_repo
    try:
        service = _new_service(client)
        result = await service.handle_user_request_async(
            "Thanks, that helps.",
            user_id="stage2-summary-user",
            session_id="stage2-summary-conversation",
            conversation_history=[],
        )
        while service._background_tasks:
            tasks = list(service._background_tasks)
            await asyncio.gather(*tasks)
            await asyncio.sleep(0)
    finally:
        business_repositories.usage_repository = original_usage
        business_repositories.conversation_repository = original_conversation
    events = [event for flush in usage_repo.flushes for event in flush["events"]]
    return _workload_record(
        workload_id="background_conversation_summary",
        title="Background conversation summary",
        execution_level="service-level",
        entrypoint="MetaRecService.handle_user_request_async → _apply_conversation_summary",
        fixture_id="eight-turn-summary-trigger-v1",
        client=client,
        events=events,
        result_summary={
            "type": result.get("type"),
            "summary_update_persisted": bool(conversation_repo.summary_updates),
            "summary_watermark": (
                conversation_repo.summary_updates[0]["watermark_id"]
                if conversation_repo.summary_updates else None
            ),
        },
        extra={
            "usage_flush_count": len(usage_repo.flushes),
            "usage_flush_scopes": [
                "request" if any(call["call_site"] == "intention.analyze" for call in client.completions.calls) else "unknown",
                "background" if any(call["call_site"] == "conversation.summary" for call in client.completions.calls) else "unknown",
            ],
        },
    )


async def _sparse_generic_task() -> Dict[str, Any]:
    client = FakeClient(
        async_responses={
            "generic.gather_reasoner": _response(_json({"action": "stop"}), 140, 25),
        },
        scope_by_label={"generic.gather_reasoner": "task"},
    )
    registry = _fake_attraction_registry()
    ledger = llm_usage.UsageLedger()
    token = llm_usage.push_ledger(ledger)
    try:
        async def reasoner(context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            return await propose_gather_action(
                client,
                query=context.get("query", ""),
                domain=context.get("domain", "attraction"),
                preferences=context.get("preferences", {}),
                observations=context.get("observations", []),
                tools=context.get("tools", []),
                found=context.get("found", 0),
                target=context.get("target", 0),
                model=MODEL,
            )

        result = await run_generic_domain_graph(
            query="Find a few things to do in Sentosa",
            domain="attraction",
            preferences={"location": "Sentosa", "attraction_types": ["museum"]},
            use_online_agent=False,
            tool_tags=["#place", "#attraction"],
            adapters=GenericGraphAdapters(tool_registry=registry, reasoner=reasoner),
        )
    finally:
        llm_usage.reset_ledger(token)
    return _workload_record(
        workload_id="sparse_generic_recommendation_task",
        title="Sparse generic recommendation task",
        execution_level="graph-level",
        entrypoint="run_generic_domain_graph",
        fixture_id="one-candidate-stop-reasoner-v1",
        client=client,
        events=list(ledger.events),
        result_summary={
            "domain": result.metadata.get("domain"),
            "selected_tools": result.metadata.get("selected_tools"),
            "candidate_count": result.metadata.get("candidate_count"),
            "gather_iterations": result.metadata.get("gather_iterations"),
            "items_count": len(result.items),
        },
        extra={"external_tool_calls": 0},
    )


async def _restaurant_planning_and_summary() -> Dict[str, Any]:
    client = FakeClient(
        channel="sync",
        sync_responses={
            "restaurant.plan": _response(
                None,
                220,
                45,
                tool_calls=[
                    {
                        "id": "call-fixture-gmap",
                        "type": "function",
                        "function": {
                            "name": "gmap.search",
                            "arguments": _json({"query": "Singapore Bugis spicy casual restaurants"}),
                        },
                    }
                ],
            ),
            "restaurant.summary": _response(
                _json(
                    {
                        "recommendations": [
                            {
                                "name": "Fixture Bistro",
                                "area": "Bugis",
                                "cuisine": "Asian",
                                "type": "casual",
                                "price_per_person_sgd": "20-30",
                                "rating": 4.7,
                                "reviews_count": 220,
                                "why": "Matches the fixture preferences.",
                            }
                        ]
                    }
                ),
                260,
                75,
            ),
        },
        scope_by_label={"restaurant.plan": "task", "restaurant.summary": "task"},
    )
    ledger = llm_usage.UsageLedger()
    token = llm_usage.push_ledger(ledger)
    try:
        result = await run_restaurant_graph(
            client=client,
            summary_model=MODEL,
            planning_model=MODEL,
            query="Recommend spicy casual restaurants in Bugis",
            preferences={"location": "Bugis", "flavor_profiles": ["spicy"]},
            user_input='{"Location (Singapore)":"Bugis","Flavor Profile":"Spicy"}',
            use_online_agent=True,
            tool_tags=["#place", "#restaurant"],
            adapters=RestaurantGraphAdapters(tool_registry=_fake_restaurant_registry()),
        )
    finally:
        llm_usage.reset_ledger(token)
    return _workload_record(
        workload_id="restaurant_planning_and_summary",
        title="Restaurant planning plus summarization",
        execution_level="graph-level",
        entrypoint="run_restaurant_graph",
        fixture_id="single-gmap-success-v1",
        client=client,
        events=list(ledger.events),
        result_summary={
            "graph": result.metadata.get("graph"),
            "plan_calls": len(result.plan_calls),
            "executions": len(result.executions),
            "checked_restaurants": len(result.checked_restaurants),
            "refine_used": result.refine_used,
        },
        extra={"external_tool_calls": 0},
    )


async def _initial_itinerary_flow() -> Dict[str, Any]:
    client = FakeClient(
        async_responses={
            "intention.analyze": _response(
                _intent_payload(
                    intent="query",
                    preferences={"domain": "itinerary", "location": "Sentosa"},
                ),
                120,
                30,
            ),
            "itinerary.constraints": _response(
                _json(
                    {
                        "location": "Sentosa",
                        "horizon_days": 1,
                        "budget_mode": "unlimited",
                        "timezone": "Asia/Singapore",
                        "style": "sightseeing",
                        "pace": "balanced",
                    }
                ),
                220,
                50,
            ),
        },
        scope_by_label={"intention.analyze": "request", "itinerary.constraints": "request"},
    )
    usage_repo = FakeUsageRepository()
    original_usage = business_repositories.usage_repository
    business_repositories.usage_repository = usage_repo
    try:
        service = _new_service(client)
        result = await service.handle_user_request_async(
            "Plan a sightseeing day in Sentosa",
            user_id="stage2-itinerary-user",
            session_id=None,
            conversation_history=[],
        )
    finally:
        business_repositories.usage_repository = original_usage
    events = [event for flush in usage_repo.flushes for event in flush["events"]]
    confirmation = result.get("confirmation_request")
    return _workload_record(
        workload_id="initial_itinerary_constraint_confirmation",
        title="Initial itinerary constraint extraction plus confirmation",
        execution_level="service-level",
        entrypoint="MetaRecService.handle_user_request_async → request orchestrator",
        fixture_id="unanchored-sentosa-confirmation-v1",
        client=client,
        events=events,
        result_summary={
            "type": result.get("type"),
            "routing_mode": (result.get("routing") or {}).get("mode"),
            "confirmation_present": confirmation is not None,
            "task_created": "task_id" in result,
        },
        extra={"usage_flush_count": len(usage_repo.flushes), "complete_itinerary_execution": False},
    )


def _metadata() -> Dict[str, Any]:
    tracked_inputs = (
        "client.py",
        "llm_compat.py",
        "llm_usage.py",
        "llm_service.py",
        "service.py",
        "conversation_context.py",
        "langgraph_metarec/graphs/request_orchestrator.py",
        "langgraph_metarec/graphs/generic_graph.py",
        "langgraph_metarec/graphs/restaurant_graph.py",
    )
    source_sha256 = {}
    for relative in tracked_inputs:
        path = BACKEND_ROOT / relative
        source_sha256[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BACKEND_ROOT.parent, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"], cwd=BACKEND_ROOT.parent, text=True
        ).splitlines()
    except Exception as exc:  # pragma: no cover - metadata fallback
        revision = None
        status = [f"metadata error: {type(exc).__name__}: {exc}"]
    safe_environment = {
        name: os.getenv(name)
        for name in (
            "LLM_MAX_FORMAT_RETRIES",
            "LLM_TEXT_MAX_TOKENS",
            "METAREC_CONTEXT_WINDOW_TURNS",
            "METAREC_CONTEXT_SUMMARY_TRIGGER",
            "ITINERARY_SOLVER",
            "ITINERARY_MAX_PROVIDER_CALLS",
            "ITINERARY_MAX_RETRIEVAL_ROUNDS",
            "METAREC_CHECKPOINTER_BACKEND",
        )
        if os.getenv(name) is not None
    }
    return {
        "experiment": "phase1-stage2-representative-workload-baseline",
        "schema_version": "stage2-representative-workload.v1",
        "git_revision": revision,
        "git_status_short": status,
        "source_sha256": source_sha256,
        "python": sys.version,
        "safe_environment": safe_environment,
        "synthetic_price_table": PRICE_TABLE,
        "live_provider_calls": False,
        "external_api_calls": False,
        "latency_measured": False,
    }


async def _run() -> Dict[str, Any]:
    os.environ["LLM_PRICE_TABLE_JSON"] = json.dumps(PRICE_TABLE, separators=(",", ":"))
    workloads = [
        await _normal_recommendation(),
        await _background_summary(),
        await _sparse_generic_task(),
        await _restaurant_planning_and_summary(),
        await _initial_itinerary_flow(),
    ]
    aggregate_events = []
    for workload in workloads:
        aggregate_events.extend(workload["usage_events"])
    return {
        "metadata": _metadata(),
        "workloads": workloads,
        "aggregate": {
            "workload_count": len(workloads),
            "logical_llm_requests": sum(item["logical_llm_requests"] for item in workloads),
            "physical_attempts": sum(item["physical_attempts"] for item in workloads),
            "visible_usage_events": sum(item["visible_usage_events"] for item in workloads),
            "invisible_attempts": sum(item["invisible_attempts"] for item in workloads),
            "prompt_tokens": sum(item["usage"]["prompt_tokens"] for item in workloads),
            "completion_tokens": sum(item["usage"]["completion_tokens"] for item in workloads),
            "total_tokens": sum(item["usage"]["total_tokens"] for item in workloads),
            "cost_usd": round(sum(item["usage"]["cost_usd"] for item in workloads), 6),
        },
    }


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else BACKEND_ROOT / "artifacts" / "stage2_representative_workload_raw.json"
    payload = asyncio.run(_run())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output} ({len(payload['workloads'])} workloads)")


if __name__ == "__main__":
    main()
