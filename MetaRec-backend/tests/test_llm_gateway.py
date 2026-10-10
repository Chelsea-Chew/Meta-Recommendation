"""Unit test suite for LLM Gateway, Provider Registry, Fake Adapter, and Budgets (WP5 Member A)."""

from __future__ import annotations

import pytest
import time

from llm_gateway import (
    AttemptOutcome,
    BudgetConfigV1,
    BudgetExhaustedError,
    CompatibilityError,
    GenerationConfigV1,
    LLMRequestV1,
    MessageV1,
    PolicyContextV1,
    PolicyLeg,
    PricingSnapshotV1,
    ProviderCapabilityV1,
    RequestMode,
    RequestRole,
    ResponseContractV1,
    ResponseKind,
    ResultStatus,
    RouteV1,
    RoutingMode,
    RoutingPolicyV1,
    ScopedToolCatalogV1,
    ToolSpecV1,
    TransportError,
    get_gateway_engine,
    get_provider_registry,
    invoke_llm,
    invoke_llm_async,
    is_kill_switch_active,
    set_kill_switch,
)
from llm_gateway.adapters.fake import FakeLLMAdapter, ScriptedResponse
from llm_usage import UsageLedger, push_ledger, reset_ledger

pytestmark = pytest.mark.backend_unit


@pytest.fixture
def fake_adapter() -> FakeLLMAdapter:
    adapter = FakeLLMAdapter()
    engine = get_gateway_engine()
    engine.register_adapter("fake_provider", adapter)
    set_kill_switch(False)
    yield adapter
    adapter.clear()
    set_kill_switch(False)


def _make_request(
    *,
    policy_tag: str = "conversation.summary",
    messages: list[MessageV1] | None = None,
    budget: BudgetConfigV1 | None = None,
    response_contract: ResponseContractV1 | None = None,
    routing_policy: RoutingPolicyV1 | None = None,
    policy_context: PolicyContextV1 | None = None,
    tools: ScopedToolCatalogV1 | None = None,
    tool_choice: str | None = None,
) -> LLMRequestV1:
    default_routing = RoutingPolicyV1(
        mode=RoutingMode.BASELINE,
        baseline_route=RouteV1(provider_id="fake_provider", model_id="fake-strong", policy_leg=PolicyLeg.BASELINE),
        cheap_route=RouteV1(provider_id="fake_provider", model_id="fake-fast", policy_leg=PolicyLeg.CHEAP),
    )
    return LLMRequestV1(
        request_id="req-test-123",
        policy_tag=policy_tag,
        messages=messages or [MessageV1(role=RequestRole.USER, content="Hello")],
        generation=GenerationConfigV1(temperature=0.7, max_output_tokens=100),
        response_contract=response_contract or ResponseContractV1(kind=ResponseKind.TEXT),
        routing_policy=routing_policy or default_routing,
        budget=budget or BudgetConfigV1(deadline_ms=5000, max_physical_attempts=3, max_cost_usd=0.50),
        policy_context=policy_context or PolicyContextV1(contains_itinerary_state=False, request_mode=RequestMode.SINGLE_DOMAIN),
        tools=tools,
        tool_choice=tool_choice,
    )


def test_provider_registry_contains_approved_sources():
    registry = get_provider_registry()
    fake_prov = registry.get_provider("fake_provider")
    groq_prov = registry.get_provider("groq")
    azure_prov = registry.get_provider("azure_openai")

    assert fake_prov is not None
    assert fake_prov.enabled is True
    assert len(fake_prov.models) >= 2

    assert groq_prov is not None
    assert azure_prov is not None

    pricing = registry.get_pricing_snapshot("default-pricing-v1")
    assert pricing.currency == "USD"
    cost = pricing.calculate_cost_usd("fake_provider", "fake-strong", 1000, 500)
    assert cost > 0.0


def test_sync_invoke_llm_success_text(fake_adapter: FakeLLMAdapter):
    fake_adapter.enqueue_response(
        ScriptedResponse(text="Deterministic test response", prompt_tokens=20, completion_tokens=10)
    )

    req = _make_request()
    result = invoke_llm(req)

    assert result.status == ResultStatus.SUCCEEDED
    assert result.output is not None
    assert result.output.text == "Deterministic test response"
    assert len(result.attempts) == 1
    assert result.attempts[0].outcome == AttemptOutcome.SUCCESS
    assert result.usage.prompt_tokens == 20
    assert result.usage.completion_tokens == 10
    assert result.usage.cost_usd > 0.0


@pytest.mark.asyncio
async def test_async_invoke_llm_success_text(fake_adapter: FakeLLMAdapter):
    fake_adapter.enqueue_response(
        ScriptedResponse(text="Async deterministic response", prompt_tokens=30, completion_tokens=15)
    )

    req = _make_request()
    result = await invoke_llm_async(req)

    assert result.status == ResultStatus.SUCCEEDED
    assert result.output is not None
    assert result.output.text == "Async deterministic response"
    assert len(result.attempts) == 1
    assert result.attempts[0].outcome == AttemptOutcome.SUCCESS


def test_sync_invoke_llm_json_validation_success(fake_adapter: FakeLLMAdapter):
    fake_adapter.enqueue_response(
        ScriptedResponse(text='{"status": "ok", "items_count": 5}', prompt_tokens=15, completion_tokens=10)
    )

    contract = ResponseContractV1(
        kind=ResponseKind.JSON,
        schema_id="test.json.v1",
        json_schema={
            "type": "object",
            "required": ["status", "items_count"],
            "properties": {"status": {"type": "string"}, "items_count": {"type": "integer"}},
        },
    )
    req = _make_request(response_contract=contract)
    result = invoke_llm(req)

    assert result.status == ResultStatus.SUCCEEDED
    assert result.output is not None
    assert result.output.json_payload == {"status": "ok", "items_count": 5}
    assert result.validation.json_parse.value == "passed"
    assert result.validation.json_schema.value == "passed"


def test_sync_invoke_llm_json_schema_failure(fake_adapter: FakeLLMAdapter):
    # Enqueue response missing required "items_count"
    fake_adapter.enqueue_response(
        ScriptedResponse(text='{"status": "ok"}', prompt_tokens=15, completion_tokens=10)
    )

    contract = ResponseContractV1(
        kind=ResponseKind.JSON,
        schema_id="test.json.v1",
        json_schema={
            "type": "object",
            "required": ["status", "items_count"],
            "properties": {"status": {"type": "string"}, "items_count": {"type": "integer"}},
        },
    )
    req = _make_request(response_contract=contract)
    result = invoke_llm(req)

    assert result.status == ResultStatus.FAILED
    assert result.validation.json_schema.value == "failed"
    assert len(result.validation.errors) > 0
    assert result.attempts[0].outcome == AttemptOutcome.VALIDATION_ERROR


def test_transport_retry_and_recovery(fake_adapter: FakeLLMAdapter):
    # First attempt: 429 RateLimit error, Second attempt: Success
    fake_adapter.enqueue_response(
        ScriptedResponse(error_to_raise=TransportError("rate_limited", "Rate limit hit", retryable=True))
    )
    fake_adapter.enqueue_response(
        ScriptedResponse(text="Recovered after rate limit", prompt_tokens=25, completion_tokens=12)
    )

    req = _make_request()
    result = invoke_llm(req)

    assert result.status == ResultStatus.SUCCEEDED
    assert result.output.text == "Recovered after rate limit"
    assert len(result.attempts) == 2
    assert result.attempts[0].outcome == AttemptOutcome.TRANSPORT_ERROR
    assert result.attempts[1].outcome == AttemptOutcome.SUCCESS


def test_compatibility_retry_removes_unsupported_temperature(fake_adapter: FakeLLMAdapter):
    # First attempt: raises Unsupported temperature error, Second attempt: succeeds
    fake_adapter.enqueue_response(
        ScriptedResponse(error_to_raise=CompatibilityError("unsupported_parameter", "temperature is not supported"))
    )
    fake_adapter.enqueue_response(
        ScriptedResponse(text="Temperature adapted response", prompt_tokens=20, completion_tokens=10)
    )

    req = _make_request()
    result = invoke_llm(req)

    assert result.status == ResultStatus.SUCCEEDED
    assert len(result.attempts) == 2
    assert result.attempts[0].outcome == AttemptOutcome.COMPATIBILITY_ERROR
    assert result.attempts[1].outcome == AttemptOutcome.SUCCESS
    assert "temperature" in result.attempts[1].applied_compatibility_changes


def test_budget_exhaustion_attempt_limit(fake_adapter: FakeLLMAdapter):
    # Continuously fail with 500 error
    fake_adapter.enqueue_response(ScriptedResponse(error_to_raise=TransportError("provider_5xx", "Server Error 500")))
    fake_adapter.enqueue_response(ScriptedResponse(error_to_raise=TransportError("provider_5xx", "Server Error 500")))

    # Set budget to max 2 attempts
    budget = BudgetConfigV1(deadline_ms=5000, max_physical_attempts=2, max_cost_usd=0.50)
    req = _make_request(budget=budget)
    result = invoke_llm(req)

    assert result.status == ResultStatus.FAILED
    assert len(result.attempts) == 2
    assert result.error is not None
    assert "budget" in result.error["error_class"]


def test_budget_exhaustion_max_cost(fake_adapter: FakeLLMAdapter):
    # First attempt generates cost of $0.05
    fake_adapter.enqueue_response(
        ScriptedResponse(prompt_tokens=10000, completion_tokens=5000, error_to_raise=TransportError("timeout", "Timeout"))
    )
    # Set tight max_cost_usd of $0.01
    budget = BudgetConfigV1(deadline_ms=5000, max_physical_attempts=5, max_cost_usd=0.01)
    req = _make_request(budget=budget)
    result = invoke_llm(req)

    assert result.status == ResultStatus.FAILED
    assert result.error is not None


def test_cheap_route_selection_and_fallback_ladder(fake_adapter: FakeLLMAdapter):
    # Set routing mode to cheap_first
    routing = RoutingPolicyV1(
        mode=RoutingMode.CHEAP_FIRST,
        baseline_route=RouteV1(provider_id="fake_provider", model_id="fake-strong", policy_leg=PolicyLeg.BASELINE),
        cheap_route=RouteV1(provider_id="fake_provider", model_id="fake-fast", policy_leg=PolicyLeg.CHEAP),
    )
    # Eligible tag + single domain policy context
    req = _make_request(
        policy_tag="conversation.summary",
        routing_policy=routing,
        policy_context=PolicyContextV1(contains_itinerary_state=False, request_mode=RequestMode.SINGLE_DOMAIN),
    )

    # 1. Successful execution on cheap route
    fake_adapter.enqueue_response(ScriptedResponse(text="Cheap model output"))
    res_cheap = invoke_llm(req)
    assert res_cheap.status == ResultStatus.SUCCEEDED
    assert res_cheap.selected_route.model_id == "fake-fast"
    assert res_cheap.selected_route.policy_leg == PolicyLeg.CHEAP

    # 2. Ineligible safety context (contains itinerary state) -> should fall back to baseline
    req_ineligible = _make_request(
        policy_tag="conversation.summary",
        routing_policy=routing,
        policy_context=PolicyContextV1(contains_itinerary_state=True, request_mode=RequestMode.SINGLE_DOMAIN),
    )
    fake_adapter.enqueue_response(ScriptedResponse(text="Baseline model output"))
    res_baseline = invoke_llm(req_ineligible)
    assert res_baseline.status == ResultStatus.SUCCEEDED
    assert res_baseline.selected_route.model_id == "fake-strong"
    assert res_baseline.selected_route.policy_leg == PolicyLeg.BASELINE


def test_tool_calling_wire_sanitization_and_restoration(fake_adapter: FakeLLMAdapter):
    tools = ScopedToolCatalogV1(
        catalog_id="cat-01",
        tools=[
            ToolSpecV1(name="gmap.search_restaurants", description="Search restaurants on Google Maps"),
            ToolSpecV1(name="spotify.play_music", description="Play music track"),
        ],
    )
    contract = ResponseContractV1(kind=ResponseKind.TOOL_CALLS, schema_id="tool.v1")
    req = _make_request(response_contract=contract, tools=tools)

    # Adapter returns sanitized name "gmap_search_restaurants"
    fake_adapter.enqueue_response(
        ScriptedResponse(
            tool_calls=[
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "gmap_search_restaurants", "arguments": '{"query": "Pasta"}'},
                }
            ]
        )
    )

    result = invoke_llm(req)
    assert result.status == ResultStatus.SUCCEEDED
    assert result.output.kind == ResponseKind.TOOL_CALLS
    assert len(result.output.tool_calls) == 1
    # Check that wire name was restored back to dotted catalog name
    assert result.output.tool_calls[0].name == "gmap.search_restaurants"
    assert result.output.tool_calls[0].arguments == {"query": "Pasta"}


def test_kill_switch_forces_baseline(fake_adapter: FakeLLMAdapter):
    routing = RoutingPolicyV1(
        mode=RoutingMode.CHEAP_FIRST,
        baseline_route=RouteV1(provider_id="fake_provider", model_id="fake-strong", policy_leg=PolicyLeg.BASELINE),
        cheap_route=RouteV1(provider_id="fake_provider", model_id="fake-fast", policy_leg=PolicyLeg.CHEAP),
    )
    req = _make_request(
        policy_tag="conversation.summary",
        routing_policy=routing,
        policy_context=PolicyContextV1(contains_itinerary_state=False, request_mode=RequestMode.SINGLE_DOMAIN),
    )

    # Engage kill switch
    set_kill_switch(True, reason="Emergency provider outage")
    assert is_kill_switch_active() is True

    fake_adapter.enqueue_response(ScriptedResponse(text="Baseline forced response"))
    result = invoke_llm(req)

    assert result.status == ResultStatus.SUCCEEDED
    assert result.selected_route.model_id == "fake-strong"
    assert result.selected_route.policy_leg == PolicyLeg.BASELINE


def test_exactly_once_usage_accounting_in_ledger(fake_adapter: FakeLLMAdapter):
    ledger = UsageLedger()
    token = push_ledger(ledger)

    try:
        fake_adapter.enqueue_response(
            ScriptedResponse(error_to_raise=TransportError("rate_limited", "Rate limit", retryable=True))
        )
        fake_adapter.enqueue_response(
            ScriptedResponse(text="Success", prompt_tokens=100, completion_tokens=50)
        )

        req = _make_request()
        result = invoke_llm(req)

        assert result.status == ResultStatus.SUCCEEDED
        assert len(result.attempts) == 2

        # Verify ledger has exactly 2 recorded events (Attempt 1: 50+25, Attempt 2: 100+50)
        assert len(ledger.events) == 2
        assert ledger.totals["prompt_tokens"] == 150  # 50 (failed attempt) + 100 (successful attempt)
        assert ledger.totals["completion_tokens"] == 75  # 25 (failed attempt) + 50 (successful attempt)
    finally:
        reset_ledger(token)
