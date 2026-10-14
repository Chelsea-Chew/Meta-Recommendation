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
    OutputPayloadV1,
    ValidationGateStatus,
    ValidationGateResultV1,
    ValidationSummaryV1,
    create_engine,
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
    assert result.validation.json_schema == ValidationGateStatus.NOT_APPLICABLE
    assert result.validation.errors == []
    assert result.attempts[0].outcome == AttemptOutcome.VALIDATION_ERROR
    assert result.attempts[0].validation_summary.json_schema.value == "failed"
    assert len(result.attempts[0].validation_summary.errors) > 0


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


def test_json_schema_strictly_required_for_json_contract():
    with pytest.raises(ValueError, match="json_schema is strictly required"):
        ResponseContractV1(kind=ResponseKind.JSON, json_schema=None)

    with pytest.raises(ValueError, match="json_schema is strictly required"):
        ResponseContractV1(kind=ResponseKind.JSON, json_schema={})

    # Valid schema succeeds
    valid_contract = ResponseContractV1(kind=ResponseKind.JSON, json_schema={"type": "object", "properties": {"a": {"type": "string"}}})
    assert valid_contract.kind == ResponseKind.JSON


def test_policy_context_strictly_requires_fields():
    # Omitting required fields raises ValidationError
    with pytest.raises(Exception):
        PolicyContextV1()  # Missing contains_itinerary_state and request_mode

    # Explicit values succeed
    ctx = PolicyContextV1(contains_itinerary_state=False, request_mode=RequestMode.SINGLE_DOMAIN)
    assert ctx.contains_itinerary_state is False
    assert ctx.request_mode == RequestMode.SINGLE_DOMAIN


def test_extra_fields_forbidden_on_request():
    with pytest.raises(Exception):
        MessageV1(role=RequestRole.USER, content="hi", unauthorized_field="bad")

    with pytest.raises(Exception):
        PolicyContextV1(contains_itinerary_state=False, request_mode=RequestMode.SINGLE_DOMAIN, extra_field="bad")


def test_tool_choice_validation_with_scoped_catalog():
    catalog = ScopedToolCatalogV1(
        catalog_id="cat-1",
        tools=[ToolSpecV1(name="search_restaurants", description="search", parameters={"type": "object"})],
    )
    # Valid choices
    req1 = _make_request(tools=catalog, tool_choice="none")
    assert req1.tool_choice == "none"

    req2 = _make_request(tools=catalog, tool_choice="auto")
    assert req2.tool_choice == "auto"

    req3 = _make_request(tools=catalog, tool_choice="search_restaurants")
    assert req3.tool_choice == "search_restaurants"

    # Invalid choice not in catalog
    with pytest.raises(ValueError, match="not present in scoped catalog"):
        _make_request(tools=catalog, tool_choice="unauthorized_tool")

    # Forced choice without catalog
    with pytest.raises(ValueError, match="no tools catalog provided"):
        _make_request(tools=None, tool_choice="some_tool")


def test_attempt_audit_includes_validation_summary_and_timestamps(fake_adapter: FakeLLMAdapter):
    fake_adapter.enqueue_response(ScriptedResponse(text="Hello world", prompt_tokens=10, completion_tokens=5))
    req = _make_request()
    res = invoke_llm(req)

    assert res.status == ResultStatus.SUCCEEDED
    assert len(res.attempts) == 1
    att = res.attempts[0]
    assert att.ended_at_utc is not None
    assert att.ended_at_utc >= att.started_at_utc
    assert att.validation_summary is not None
    assert att.validation_summary.json_parse == ValidationGateStatus.NOT_APPLICABLE
    assert att.provider_response_received is True
    assert att.is_usage_reported is True


def test_external_validator_hook_sync_and_async(fake_adapter: FakeLLMAdapter):
    fake_adapter.enqueue_response(ScriptedResponse(text='{"value": 42}'))
    fake_adapter.enqueue_response(ScriptedResponse(text='{"value": 99}'))

    calls = []

    def sync_hook(request, raw_resp, sanitized_mapping):
        calls.append(("sync", raw_resp.text))
        summary = ValidationSummaryV1(json_parse=ValidationGateStatus.PASSED)
        payload = OutputPayloadV1(kind=ResponseKind.JSON, json={"custom": True})
        return ValidationGateResultV1(passed=True, summary=summary, payload=payload)

    contract = ResponseContractV1(kind=ResponseKind.JSON, json_schema={"type": "object"})
    req = _make_request(response_contract=contract)

    # 1. Sync invocation with validator_hook parameter
    res_sync = invoke_llm(req, validator_hook=sync_hook)
    assert res_sync.status == ResultStatus.SUCCEEDED
    assert res_sync.output.json_payload == {"custom": True}
    assert len(calls) == 1
    assert calls[0][0] == "sync"

    # 2. Async invocation with async validator hook
    async def async_hook(request, raw_resp, sanitized_mapping):
        calls.append(("async", raw_resp.text))
        summary = ValidationSummaryV1(json_parse=ValidationGateStatus.PASSED)
        payload = OutputPayloadV1(kind=ResponseKind.JSON, json={"async_custom": True})
        return ValidationGateResultV1(passed=True, summary=summary, payload=payload)

    import asyncio
    res_async = asyncio.run(invoke_llm_async(req, validator_hook=async_hook))
    assert res_async.status == ResultStatus.SUCCEEDED
    assert res_async.output.json_payload == {"async_custom": True}
    assert len(calls) == 2
    assert calls[1][0] == "async"


def test_validation_safeguards_output_release(fake_adapter: FakeLLMAdapter):
    # Safeguard 1: passed=True with payload=None raises ValueError
    summary = ValidationSummaryV1()
    with pytest.raises(ValueError, match="passed=True must have a non-null payload"):
        ValidationGateResultV1(passed=True, summary=summary, payload=None)

    # Safeguard 2: passed=False with payload!=None raises ValueError
    with pytest.raises(ValueError, match="passed=False must have payload=None"):
        ValidationGateResultV1(
            passed=False,
            summary=summary,
            payload=OutputPayloadV1(kind=ResponseKind.TEXT, text="bad"),
        )

    # Safeguard 3: Payload kind mismatch with contract is rejected and not released
    def mismatched_hook(request, raw_resp, sanitized_mapping):
        # Contract expects JSON, but hook returns TEXT payload
        return ValidationGateResultV1(
            passed=True,
            summary=ValidationSummaryV1(),
            payload=OutputPayloadV1(kind=ResponseKind.TEXT, text="raw text"),
        )

    contract = ResponseContractV1(kind=ResponseKind.JSON, json_schema={"type": "object"})
    req = _make_request(response_contract=contract)
    fake_adapter.enqueue_response(ScriptedResponse(text='{"status": "ok"}'))

    res = invoke_llm(req, validator_hook=mismatched_hook)
    assert res.status == ResultStatus.FAILED
    assert res.output is None
    assert res.attempts[0].outcome == AttemptOutcome.VALIDATION_ERROR
    assert any("Expected JSON payload kind" in err for err in res.attempts[0].validation_summary.errors)


def test_missing_usage_and_provider_zero_accounting(fake_adapter: FakeLLMAdapter):
    ledger = UsageLedger()
    token = push_ledger(ledger)

    try:
        # 1. Missing usage attempt (prompt_tokens=None, completion_tokens=None)
        fake_adapter.enqueue_response(
            ScriptedResponse(text="Missing usage response", prompt_tokens=None, completion_tokens=None)
        )
        req1 = _make_request()
        res1 = invoke_llm(req1)

        assert res1.status == ResultStatus.SUCCEEDED
        att1 = res1.attempts[0]
        assert att1.is_usage_reported is False
        assert att1.prompt_tokens is None
        assert att1.completion_tokens is None
        assert att1.cost_usd == 0.0
        assert res1.usage.is_estimated is False
        assert res1.usage.cost_usd == 0.0

        # Verify ledger has 1 event with 0 tokens and $0 cost
        assert len(ledger.events) == 1
        assert ledger.events[0].prompt_tokens == 0
        assert ledger.events[0].cost_usd == 0.0

        # 2. Provider reported zero attempt (prompt_tokens=0, completion_tokens=0)
        fake_adapter.enqueue_response(
            ScriptedResponse(text="Zero usage response", prompt_tokens=0, completion_tokens=0)
        )
        req2 = _make_request()
        res2 = invoke_llm(req2)

        assert res2.status == ResultStatus.SUCCEEDED
        att2 = res2.attempts[0]
        assert att2.is_usage_reported is True
        assert att2.prompt_tokens == 0
        assert att2.completion_tokens == 0
        assert att2.cost_usd == 0.0
        assert res2.usage.is_estimated is False
        assert res2.usage.cost_usd == 0.0

        # Verify ledger now has 2 events total (both exactly-once recorded)
        assert len(ledger.events) == 2
        assert ledger.events[1].prompt_tokens == 0
        assert ledger.events[1].cost_usd == 0.0
    finally:
        reset_ledger(token)


def test_failed_result_validation_summary_separation(fake_adapter: FakeLLMAdapter):
    # Enqueue a response that fails validation
    fake_adapter.enqueue_response(
        ScriptedResponse(text='{"wrong": 123}')
    )
    contract = ResponseContractV1(
        kind=ResponseKind.JSON,
        schema_id="test.schema",
        json_schema={"type": "object", "required": ["expected_field"], "properties": {"expected_field": {"type": "string"}}},
    )
    req = _make_request(response_contract=contract)
    result = invoke_llm(req)

    assert result.status == ResultStatus.FAILED
    assert result.output is None

    # Result-level validation: NOT_APPLICABLE and errors=[]
    assert result.validation.json_parse == ValidationGateStatus.NOT_APPLICABLE
    assert result.validation.json_schema == ValidationGateStatus.NOT_APPLICABLE
    assert result.validation.semantic == ValidationGateStatus.NOT_APPLICABLE
    assert result.validation.errors == []
    assert result.validation.validator_id == "none"

    # Attempt-level validation retains candidate validation details
    assert len(result.attempts) == 1
    assert result.attempts[0].outcome == AttemptOutcome.VALIDATION_ERROR
    assert result.attempts[0].validation_summary.json_schema == ValidationGateStatus.FAILED
    assert len(result.attempts[0].validation_summary.errors) > 0

    # Structured error explains failure
    assert result.error is not None
    assert result.error.code == "validation_failed"


