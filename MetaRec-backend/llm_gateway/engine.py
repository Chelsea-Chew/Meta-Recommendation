"""Core Gateway Engine implementing sync and async multi-provider execution (v1).

Production-hardening fixes applied here
-----------------------------------------
Fix #1  – Exponential back-off with full jitter on TransportError retries.
           Replaces the flat 50 ms sleep that caused thundering-herd under load.

Fix #2  – ``invoke_llm`` (sync) is now safe to call from async callers via
           ``invoke_llm_in_executor``.  A module-level note makes the
           blocking nature of the sync path explicit.

Fix #3  – ``create_engine`` / ``create_engine_for_registry`` factory helpers
           let test code (or DI containers) create fully isolated engine
           instances instead of mutating the shared global singleton.
           The global singleton is retained for backward compat but is no
           longer the only option.

Fix #8  – ``PrivacyConfigV1.allow_prompt_logging`` is now respected.
           When set to ``False`` the engine logs ``[REDACTED]`` for message
           content and response text.  The FakeLLMAdapter still stores the
           full request in ``call_history`` (testing only); live adapters
           are not affected because they receive the same normalized_kwargs.

Fix #10 – Token budget is pre-validated against the registered model's
           ``ModelLimitV1.max_output_tokens`` before the first attempt.
           A request asking for more tokens than the model allows is
           clamped (with a warning) instead of burning one attempt on a
           guaranteed 400 error.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from llm_gateway.adapters.anthropic import AnthropicAdapter
from llm_gateway.adapters.base import BaseLLMAdapter, RawAdapterResponse
from llm_gateway.adapters.fake import FakeLLMAdapter
from llm_gateway.adapters.openai_base import OpenAIBaseAdapter
from llm_gateway.adapters.openai_compat import OpenAICompatibleAdapter
from llm_gateway.constants import (
    CHEAP_ROUTE_ELIGIBLE_TAGS,
    DEFAULT_BACKOFF_BASE_MS,
    DEFAULT_BACKOFF_CAP_MS,
    DEFAULT_MAX_ATTEMPTS_PER_ROUTE,
    DEFAULT_TOOL_NAME_PATTERN,
    AdapterKind,
    DegradationReason,
    GatewayErrorCode,
    HTTP_STATUS_INTERNAL_SERVER_ERROR,
    HTTP_STATUS_OK,
    MS_PER_SECOND,
    ProviderId,
)
from llm_gateway.contracts import (
    AttemptKind,
    AttemptOutcome,
    DegradationSummaryV1,
    LLMRequestV1,
    LLMResultV1,
    OutputPayloadV1,
    PhysicalAttemptV1,
    PolicyLeg,
    RequestMode,
    ResponseKind,
    ResultStatus,
    RouteV1,
    RoutingMode,
    TimingSummaryV1,
    ToolCallOutputV1,
    UsageSummaryV1,
    ValidationGateStatus,
    ValidationSummaryV1,
)
from llm_gateway.errors import (
    BudgetExhaustedError,
    CompatibilityError,
    ErrorClass,
    LLMErrorV1,
    OutputValidationError,
    PolicyViolationError,
    ProviderError,
    RequestValidationError,
    TransportError,
)
from llm_gateway.kill_switch import is_kill_switch_active, get_kill_switch_reason
from llm_gateway.registry import ProviderRegistry, get_provider_registry
from llm_gateway.validators import (
    get_validator_registry,
    validate_json_schema,
    validate_json_syntax,
)

logger = logging.getLogger(__name__)

_TOOL_NAME_ALLOWED_RE = re.compile(DEFAULT_TOOL_NAME_PATTERN)

# Low-risk tags eligible for cheap routing when safe
_CHEAP_ROUTE_ELIGIBLE_TAGS = CHEAP_ROUTE_ELIGIBLE_TAGS

# ── Fix #1: Retry back-off constants ────────────────────────────────────────
#   Uses "full jitter" strategy: sleep = random(0, min(cap, base * 2^attempt))
#   This distributes retry storms across the retry window instead of bunching.
_BACKOFF_BASE_MS: float = DEFAULT_BACKOFF_BASE_MS   # Starting window (ms)
_BACKOFF_CAP_MS: float = DEFAULT_BACKOFF_CAP_MS  # Maximum sleep cap (ms)
# ────────────────────────────────────────────────────────────────────────────

# Max retries per-route (was previously the magic number 2 in while < 2)
# Fix #3-note: This is a module-level named constant; tests can override via
# monkey-patch if needed, or pass a custom engine with a custom value.
_MAX_ATTEMPTS_PER_ROUTE: int = DEFAULT_MAX_ATTEMPTS_PER_ROUTE


def _jitter_sleep_seconds(attempt_number: int) -> float:
    """Compute full-jitter exponential back-off sleep duration in seconds.

    Fix #1: Replaces the flat ``time.sleep(0.05)`` used previously.
    ``attempt_number`` is 1-indexed (first retry = 1).
    """
    cap = _BACKOFF_CAP_MS
    base = _BACKOFF_BASE_MS
    window_ms = min(cap, base * math.pow(2.0, attempt_number - 1))
    sleep_ms = random.uniform(0.0, window_ms)
    return sleep_ms / MS_PER_SECOND



class GatewayEngine:
    """Execution engine for LLMRequestV1 -> LLMResultV1 lifecycle.

    Fix #3: Constructor accepts an optional ``registry`` so isolated engine
    instances can be created for testing without mutating the global singleton.
    Use ``create_engine()`` or ``create_engine_for_registry()`` factories from
    this module.
    """

    def __init__(self, registry: Optional[ProviderRegistry] = None) -> None:
        self.registry = registry or get_provider_registry()
        self._adapters: Dict[str, BaseLLMAdapter] = {
            ProviderId.FAKE.value: FakeLLMAdapter(),
            ProviderId.GROQ.value: OpenAICompatibleAdapter(provider_id=ProviderId.GROQ.value),
            ProviderId.AZURE_OPENAI.value: OpenAICompatibleAdapter(provider_id=ProviderId.AZURE_OPENAI.value),
            ProviderId.OPENAI.value: OpenAICompatibleAdapter(provider_id=ProviderId.OPENAI.value),
            ProviderId.OPENAI_BASE.value: OpenAIBaseAdapter(provider_id=ProviderId.OPENAI_BASE.value),
            ProviderId.ANTHROPIC.value: AnthropicAdapter(provider_id=ProviderId.ANTHROPIC.value),
        }

    def register_adapter(self, provider_id: str, adapter: BaseLLMAdapter) -> None:
        self._adapters[provider_id] = adapter

    def get_adapter(self, provider_id: str) -> BaseLLMAdapter:
        if provider_id in self._adapters:
            return self._adapters[provider_id]

        prov_cap = self.registry.get_provider(provider_id)
        if prov_cap:
            if prov_cap.adapter_kind == AdapterKind.ANTHROPIC.value:
                adapter = AnthropicAdapter(provider_id=provider_id)
                self._adapters[provider_id] = adapter
                return adapter
            if prov_cap.adapter_kind in (AdapterKind.OPENAI_BASE.value, AdapterKind.OPENAI_HTTP.value):
                adapter = OpenAIBaseAdapter(provider_id=provider_id)
                self._adapters[provider_id] = adapter
                return adapter

        # Default to OpenAI compatible adapter
        adapter = OpenAICompatibleAdapter(provider_id=provider_id)
        self._adapters[provider_id] = adapter
        return adapter

    def _determine_initial_route(self, request: LLMRequestV1) -> Tuple[RouteV1, PolicyLeg]:
        """Determine whether request starts on cheap_route or baseline_route."""
        routing = request.routing_policy
        if is_kill_switch_active():
            logger.info("Kill switch active (%s); forcing baseline route", get_kill_switch_reason())
            return routing.baseline_route, PolicyLeg.BASELINE

        if routing.mode == RoutingMode.CHEAP_FIRST and routing.cheap_route is not None:
            # Check safety allowlist
            ctx = request.policy_context
            tag = request.policy_tag
            is_tag_eligible = tag in _CHEAP_ROUTE_ELIGIBLE_TAGS
            is_safety_ok = (not ctx.contains_itinerary_state) and (ctx.request_mode == RequestMode.SINGLE_DOMAIN)
            if is_tag_eligible and is_safety_ok:
                return routing.cheap_route, PolicyLeg.CHEAP

        return routing.baseline_route, PolicyLeg.BASELINE


    def _build_route_ladder(self, request: LLMRequestV1, initial_route: RouteV1, initial_leg: PolicyLeg) -> List[Tuple[RouteV1, PolicyLeg]]:
        """Construct ordered route fallback ladder."""
        ladder: List[Tuple[RouteV1, PolicyLeg]] = [(initial_route, initial_leg)]
        # If initial route was cheap, append baseline route as first fallback
        if initial_leg == PolicyLeg.CHEAP:
            ladder.append((request.routing_policy.baseline_route, PolicyLeg.BASELINE))
        # Append remaining declared fallback routes
        for fb in request.routing_policy.fallback_routes:
            if (fb.provider_id, fb.model_id) != (initial_route.provider_id, initial_route.model_id):
                ladder.append((fb, PolicyLeg.FALLBACK))
        return ladder

    def _clamp_output_tokens(self, request: LLMRequestV1, route: RouteV1) -> int:
        """Return the effective max_output_tokens clamped to the model limit.

        Fix #10: Previously the engine would pass the caller-requested token
        count directly to the provider even when it exceeded the model's hard
        limit, causing a guaranteed 400 error that wasted one attempt.  Now we
        pre-validate and clamp with a WARNING so the caller is informed.
        """
        requested = request.generation.max_output_tokens
        provider_cap = self.registry.get_provider(route.provider_id)
        if provider_cap:
            for m in provider_cap.models:
                if m.model_id == route.model_id:
                    limit = m.max_output_tokens
                    if requested > limit:
                        logger.warning(
                            "Request %s asks for %d output tokens but model %s/%s limit is %d; "
                            "clamping to avoid a guaranteed 400 error (Fix #10).",
                            request.request_id,
                            requested,
                            route.provider_id,
                            route.model_id,
                            limit,
                        )
                        return limit
                    break
        return requested

    def _normalize_request_kwargs(
        self, request: LLMRequestV1, route: RouteV1, adaptations: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Convert LLMRequestV1 into provider SDK kwargs with wire sanitization."""
        messages_wire = []
        for msg in request.messages:
            m_dict: Dict[str, Any] = {"role": msg.role.value, "content": msg.content}
            if msg.name:
                m_dict["name"] = msg.name
            messages_wire.append(m_dict)

        kwargs: Dict[str, Any] = {
            "model": route.model_id,
            "messages": messages_wire,
        }

        gen = request.generation
        if gen.temperature is not None and "temperature" not in adaptations.get("removed_params", []):
            kwargs["temperature"] = gen.temperature
        if gen.seed is not None and "seed" not in adaptations.get("removed_params", []):
            kwargs["seed"] = gen.seed
        if gen.stop:
            kwargs["stop"] = gen.stop

        # Fix #10: use clamped token value
        effective_max_tokens = self._clamp_output_tokens(request, route)

        # Reasoning model check or max tokens adaptation
        provider_cap = self.registry.get_provider(route.provider_id)
        is_reasoning = False
        if provider_cap:
            for m in provider_cap.models:
                if m.model_id == route.model_id and m.is_reasoning_model:
                    is_reasoning = True
                    break

        if is_reasoning or "use_max_completion_tokens" in adaptations:
            kwargs.pop("temperature", None)
            kwargs["extra_body"] = {"max_completion_tokens": effective_max_tokens}
        else:
            kwargs["max_tokens"] = effective_max_tokens

        # JSON response format
        if request.response_contract.kind == ResponseKind.JSON:
            if provider_cap and provider_cap.features.supports_json_object:
                kwargs["response_format"] = {"type": "json_object"}

        # Tools and tool choices
        sanitized_mapping: Dict[str, str] = {}
        if request.tools and request.tools.tools:
            tools_wire = []
            for t in request.tools.tools:
                orig_name = t.name
                wire_name = orig_name
                if not _TOOL_NAME_ALLOWED_RE.match(orig_name):
                    wire_name = re.sub(r"[^a-zA-Z0-9_-]", "_", orig_name)
                    sanitized_mapping[wire_name] = orig_name
                tools_wire.append(
                    {
                        "type": "function",
                        "function": {
                            "name": wire_name,
                            "description": t.description,
                            "parameters": t.parameters or {"type": "object", "properties": {}},
                        },
                    }
                )
            kwargs["tools"] = tools_wire
            if request.tool_choice:
                if isinstance(request.tool_choice, str):
                    kwargs["tool_choice"] = request.tool_choice
                elif isinstance(request.tool_choice, dict):
                    choice_func = request.tool_choice.get("function", {})
                    fn_name = choice_func.get("name", "")
                    reverse_map = {orig: san for san, orig in sanitized_mapping.items()}
                    wire_choice_name = reverse_map.get(fn_name, fn_name)
                    kwargs["tool_choice"] = {
                        "type": "function",
                        "function": {"name": wire_choice_name},
                    }

        return kwargs, sanitized_mapping

    def _record_attempt(
        self,
        request: LLMRequestV1,
        route: RouteV1,
        leg: PolicyLeg,
        attempt_number: int,
        attempt_kind: AttemptKind,
        started_at: float,
        latency_ms: float,
        outcome: AttemptOutcome,
        resp: Optional[RawAdapterResponse] = None,
        error: Optional[Exception] = None,
        applied_changes: Optional[List[str]] = None,
    ) -> PhysicalAttemptV1:
        """Create a PhysicalAttemptV1 record and record in usage ledger."""
        attempt_id = f"att-{uuid.uuid4().hex[:12]}"
        pricing = self.registry.get_pricing_snapshot(request.pricing_snapshot_id)

        p_tokens = resp.prompt_tokens if resp else getattr(error, "prompt_tokens", 0)
        c_tokens = resp.completion_tokens if resp else getattr(error, "completion_tokens", 0)
        t_tokens = resp.total_tokens if resp else (p_tokens + c_tokens)
        cost_usd = pricing.calculate_cost_usd(route.provider_id, route.model_id, p_tokens, c_tokens)

        error_code = None
        error_msg = None
        if error:
            error_code = getattr(error, "code", type(error).__name__)
            error_msg = str(error)

        attempt = PhysicalAttemptV1(
            attempt_id=attempt_id,
            request_id=request.request_id,
            attempt_number=attempt_number,
            provider_id=route.provider_id,
            model_id=route.model_id,
            policy_leg=leg,
            attempt_kind=attempt_kind,
            started_at_utc=started_at,
            latency_ms=round(latency_ms, 2),
            outcome=outcome,
            http_status=resp.http_status if resp else (HTTP_STATUS_INTERNAL_SERVER_ERROR if error else HTTP_STATUS_OK),
            provider_request_id=resp.provider_request_id if resp else None,

            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=t_tokens,
            cost_usd=cost_usd,
            price_table_version=pricing.pricing_snapshot_id,
            applied_compatibility_changes=applied_changes or [],
            error_code=error_code,
            error_message=error_msg,
        )

        # Update llm_usage ledger if present
        try:
            from llm_usage import UsageEvent, current_ledger
            ledger = current_ledger()
            if ledger is not None:
                event = UsageEvent(
                    model=route.model_id,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    total_tokens=t_tokens,
                    cost_usd=cost_usd,
                )
                ledger.add(event)
        except Exception as exc:
            logger.debug("Could not record to llm_usage ledger: %s", exc)

        return attempt

    def _log_privacy_safe(self, request: LLMRequestV1, label: str, content: str) -> None:
        """Log content only when prompt logging is permitted by the privacy policy.

        Fix #8: Respects ``PrivacyConfigV1.allow_prompt_logging``.  When set to
        ``False`` the content is replaced with ``[REDACTED]`` in the log output.
        """
        if request.privacy.allow_prompt_logging:
            logger.debug("%s [req=%s]: %s", label, request.request_id, content)
        else:
            logger.debug("%s [req=%s]: [REDACTED – prompt logging disabled]", label, request.request_id)

    def _validate_output(
        self,
        request: LLMRequestV1,
        resp: RawAdapterResponse,
        sanitized_mapping: Dict[str, str],
    ) -> Tuple[ValidationSummaryV1, Optional[OutputPayloadV1]]:
        """Validate raw response against JSON, JSON Schema, and Semantic Validators."""
        contract = request.response_contract
        val_summary = ValidationSummaryV1(validator_id=contract.semantic_validator_id)
        raw_text = resp.text or ""

        # Handle TEXT response
        if contract.kind == ResponseKind.TEXT:
            if not raw_text and not contract.allow_empty:
                val_summary.errors.append("Empty text response not allowed")
                return val_summary, None
            output = OutputPayloadV1(kind=ResponseKind.TEXT, text=raw_text)
            return val_summary, output

        # Handle TOOL_CALLS response
        if contract.kind == ResponseKind.TOOL_CALLS:
            out_tool_calls: List[ToolCallOutputV1] = []
            for tc in resp.tool_calls:
                fn = tc.get("function", {})
                wire_name = fn.get("name", "")
                orig_name = sanitized_mapping.get(wire_name, wire_name)

                # Validate tool presence in scoped catalog
                if request.tools:
                    catalog_tool_names = {t.name for t in request.tools.tools}
                    if orig_name not in catalog_tool_names:
                        val_summary.errors.append(f"Tool '{orig_name}' not in scoped catalog")
                        val_summary.semantic = ValidationGateStatus.FAILED
                        return val_summary, None

                # Parse arguments
                raw_args = fn.get("arguments", "{}")
                try:
                    parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception as e:
                    val_summary.errors.append(f"Malformed tool arguments JSON: {e}")
                    val_summary.json_parse = ValidationGateStatus.FAILED
                    return val_summary, None

                out_tool_calls.append(
                    ToolCallOutputV1(id=tc.get("id", "call_1"), name=orig_name, arguments=parsed_args)
                )

            val_summary.json_parse = ValidationGateStatus.PASSED
            val_summary.semantic = ValidationGateStatus.PASSED
            output = OutputPayloadV1(kind=ResponseKind.TOOL_CALLS, text=raw_text, tool_calls=out_tool_calls)
            return val_summary, output

        # Handle JSON response
        if contract.kind == ResponseKind.JSON:
            ok, parsed_json, err = validate_json_syntax(raw_text)
            if not ok:
                val_summary.json_parse = ValidationGateStatus.FAILED
                val_summary.errors.append(err or "Malformed JSON")
                return val_summary, None
            val_summary.json_parse = ValidationGateStatus.PASSED

            # JSON Schema Validation (Fix #1: now full Draft-7 via jsonschema)
            schema_ok, schema_errs = validate_json_schema(parsed_json, contract.json_schema)
            if not schema_ok:
                val_summary.json_schema = ValidationGateStatus.FAILED
                val_summary.errors.extend(schema_errs)
                return val_summary, None
            val_summary.json_schema = ValidationGateStatus.PASSED

            # Semantic Validation
            sem_validator = get_validator_registry().get_validator(contract.semantic_validator_id)
            if sem_validator:
                sem_ok, sem_errs = sem_validator(parsed_json, contract.validator_inputs)
                if not sem_ok:
                    val_summary.semantic = ValidationGateStatus.FAILED
                    val_summary.errors.extend(sem_errs)
                    return val_summary, None
                val_summary.semantic = ValidationGateStatus.PASSED

            output = OutputPayloadV1(kind=ResponseKind.JSON, text=raw_text, json=parsed_json)
            return val_summary, output

        return val_summary, None

    # ── Core execution: shared attempt-loop logic ────────────────────────────

    def _run_attempt_loop_sync(
        self,
        request: LLMRequestV1,
        route_ladder: List[Tuple[RouteV1, PolicyLeg]],
        initial_route: RouteV1,
        initial_leg: PolicyLeg,
        start_time_monotonic: float,
    ) -> Tuple[
        List[PhysicalAttemptV1],
        Optional[OutputPayloadV1],
        Optional[RouteV1],
        ValidationSummaryV1,
        Optional[DegradationSummaryV1],
        Optional[LLMErrorV1],
    ]:
        """Synchronous attempt-loop body shared by invoke_llm."""
        deadline_ms = request.budget.deadline_ms
        attempts: List[PhysicalAttemptV1] = []
        adaptations: Dict[str, Any] = {"removed_params": []}

        attempt_count = 0
        last_error: Optional[LLMErrorV1] = None
        selected_route: Optional[RouteV1] = None
        released_output: Optional[OutputPayloadV1] = None
        final_val_summary = ValidationSummaryV1()
        degradation: Optional[DegradationSummaryV1] = None

        for route, leg in route_ladder:
            if released_output is not None:
                break

            route_attempts = 0
            while route_attempts < _MAX_ATTEMPTS_PER_ROUTE:
                elapsed_ms = (time.monotonic() - start_time_monotonic) * MS_PER_SECOND
                if elapsed_ms >= deadline_ms:
                    last_error = BudgetExhaustedError(GatewayErrorCode.DEADLINE_EXHAUSTED.value, f"Request deadline of {deadline_ms}ms exceeded")
                    break

                if attempt_count >= request.budget.max_physical_attempts:
                    last_error = BudgetExhaustedError(
                        GatewayErrorCode.ATTEMPT_BUDGET_EXHAUSTED.value,
                        f"Max physical attempts ({request.budget.max_physical_attempts}) reached",
                    )
                    break

                total_cost_so_far = sum(a.cost_usd for a in attempts)
                if total_cost_so_far >= request.budget.max_cost_usd:
                    last_error = BudgetExhaustedError(
                        GatewayErrorCode.COST_BUDGET_EXHAUSTED.value, f"Max cost budget ${request.budget.max_cost_usd} exceeded"
                    )
                    break

                attempt_count += 1
                route_attempts += 1
                att_kind = AttemptKind.INITIAL if attempt_count == 1 else (
                    AttemptKind.ROUTE_FALLBACK if route_attempts == 1 else AttemptKind.TRANSPORT_RETRY
                )

                att_start_utc = time.time()
                att_start_mono = time.monotonic()
                adapter = self.get_adapter(route.provider_id)
                kwargs, san_mapping = self._normalize_request_kwargs(request, route, adaptations)

                # Fix #8: log prompt content only when privacy policy allows
                self._log_privacy_safe(request, "PROMPT", str([m["content"] for m in kwargs.get("messages", [])]))

                try:
                    raw_resp = adapter.execute(request, route, kwargs)
                    lat_ms = (time.monotonic() - att_start_mono) * MS_PER_SECOND

                    # Fix #8: log response only when allowed
                    self._log_privacy_safe(request, "RESPONSE", raw_resp.text or "")

                    val_summary, out_payload = self._validate_output(request, raw_resp, san_mapping)
                    final_val_summary = val_summary

                    if out_payload is not None:
                        attempt = self._record_attempt(
                            request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                            AttemptOutcome.SUCCESS, resp=raw_resp, applied_changes=adaptations.get("removed_params")
                        )
                        attempts.append(attempt)
                        selected_route = route
                        released_output = out_payload
                        if leg != initial_leg:
                            degradation = DegradationSummaryV1(
                                reason_code=DegradationReason.FALLBACK_ROUTE_USED.value, from_route=initial_route, to_route=route
                            )
                        break
                    else:
                        val_err = OutputValidationError(GatewayErrorCode.VALIDATION_FAILED.value, f"Validation errors: {val_summary.errors}")
                        attempt = self._record_attempt(
                            request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                            AttemptOutcome.VALIDATION_ERROR, resp=raw_resp, error=val_err
                        )
                        attempts.append(attempt)
                        last_error = val_err
                        break  # Fall back to next route in ladder

                except CompatibilityError as exc:
                    lat_ms = (time.monotonic() - att_start_mono) * MS_PER_SECOND
                    attempt = self._record_attempt(
                        request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                        AttemptOutcome.COMPATIBILITY_ERROR, error=exc
                    )
                    attempts.append(attempt)
                    last_error = exc
                    if "temperature" in str(exc).lower():
                        adaptations["removed_params"].append("temperature")
                    elif "max_tokens" in str(exc).lower() or "extra_body" in str(exc).lower():
                        adaptations["use_max_completion_tokens"] = True
                    else:
                        break

                except TransportError as exc:
                    lat_ms = (time.monotonic() - att_start_mono) * MS_PER_SECOND
                    attempt = self._record_attempt(
                        request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                        AttemptOutcome.TRANSPORT_ERROR, error=exc
                    )
                    attempts.append(attempt)
                    last_error = exc
                    # Fix #1: exponential back-off with full jitter
                    sleep_sec = _jitter_sleep_seconds(route_attempts)
                    logger.debug(
                        "TransportError on attempt %d/%d; sleeping %.0f ms before retry (Fix #1 jitter).",
                        attempt_count,
                        request.budget.max_physical_attempts,
                        sleep_sec * MS_PER_SECOND,
                    )
                    time.sleep(sleep_sec)

                except Exception as exc:
                    lat_ms = (time.monotonic() - att_start_mono) * MS_PER_SECOND
                    pe = ProviderError(GatewayErrorCode.PROVIDER_ERROR.value, str(exc))
                    attempt = self._record_attempt(

                        request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                        AttemptOutcome.PROVIDER_ERROR, error=pe
                    )
                    attempts.append(attempt)
                    last_error = pe
                    break

        return attempts, released_output, selected_route, final_val_summary, degradation, last_error

    async def _run_attempt_loop_async(
        self,
        request: LLMRequestV1,
        route_ladder: List[Tuple[RouteV1, PolicyLeg]],
        initial_route: RouteV1,
        initial_leg: PolicyLeg,
        start_time_monotonic: float,
    ) -> Tuple[
        List[PhysicalAttemptV1],
        Optional[OutputPayloadV1],
        Optional[RouteV1],
        ValidationSummaryV1,
        Optional[DegradationSummaryV1],
        Optional[LLMErrorV1],
    ]:
        """Asynchronous attempt-loop body shared by invoke_llm_async."""
        deadline_ms = request.budget.deadline_ms
        attempts: List[PhysicalAttemptV1] = []
        adaptations: Dict[str, Any] = {"removed_params": []}

        attempt_count = 0
        last_error: Optional[LLMErrorV1] = None
        selected_route: Optional[RouteV1] = None
        released_output: Optional[OutputPayloadV1] = None
        final_val_summary = ValidationSummaryV1()
        degradation: Optional[DegradationSummaryV1] = None

        for route, leg in route_ladder:
            if released_output is not None:
                break

            route_attempts = 0
            while route_attempts < _MAX_ATTEMPTS_PER_ROUTE:
                elapsed_ms = (time.monotonic() - start_time_monotonic) * MS_PER_SECOND
                if elapsed_ms >= deadline_ms:
                    last_error = BudgetExhaustedError(GatewayErrorCode.DEADLINE_EXHAUSTED.value, f"Request deadline of {deadline_ms}ms exceeded")
                    break

                if attempt_count >= request.budget.max_physical_attempts:
                    last_error = BudgetExhaustedError(
                        GatewayErrorCode.ATTEMPT_BUDGET_EXHAUSTED.value,
                        f"Max physical attempts ({request.budget.max_physical_attempts}) reached",
                    )
                    break

                total_cost_so_far = sum(a.cost_usd for a in attempts)
                if total_cost_so_far >= request.budget.max_cost_usd:
                    last_error = BudgetExhaustedError(
                        GatewayErrorCode.COST_BUDGET_EXHAUSTED.value, f"Max cost budget ${request.budget.max_cost_usd} exceeded"
                    )
                    break

                attempt_count += 1
                route_attempts += 1
                att_kind = AttemptKind.INITIAL if attempt_count == 1 else (
                    AttemptKind.ROUTE_FALLBACK if route_attempts == 1 else AttemptKind.TRANSPORT_RETRY
                )

                att_start_utc = time.time()
                att_start_mono = time.monotonic()
                adapter = self.get_adapter(route.provider_id)
                kwargs, san_mapping = self._normalize_request_kwargs(request, route, adaptations)

                self._log_privacy_safe(request, "PROMPT", str([m["content"] for m in kwargs.get("messages", [])]))

                try:
                    raw_resp = await adapter.execute_async(request, route, kwargs)
                    lat_ms = (time.monotonic() - att_start_mono) * MS_PER_SECOND

                    self._log_privacy_safe(request, "RESPONSE", raw_resp.text or "")

                    val_summary, out_payload = self._validate_output(request, raw_resp, san_mapping)
                    final_val_summary = val_summary

                    if out_payload is not None:
                        attempt = self._record_attempt(
                            request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                            AttemptOutcome.SUCCESS, resp=raw_resp, applied_changes=adaptations.get("removed_params")
                        )
                        attempts.append(attempt)
                        selected_route = route
                        released_output = out_payload
                        if leg != initial_leg:
                            degradation = DegradationSummaryV1(
                                reason_code=DegradationReason.FALLBACK_ROUTE_USED.value, from_route=initial_route, to_route=route
                            )
                        break
                    else:
                        val_err = OutputValidationError(GatewayErrorCode.VALIDATION_FAILED.value, f"Validation errors: {val_summary.errors}")
                        attempt = self._record_attempt(
                            request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                            AttemptOutcome.VALIDATION_ERROR, resp=raw_resp, error=val_err
                        )
                        attempts.append(attempt)
                        last_error = val_err
                        break

                except CompatibilityError as exc:
                    lat_ms = (time.monotonic() - att_start_mono) * MS_PER_SECOND
                    attempt = self._record_attempt(
                        request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                        AttemptOutcome.COMPATIBILITY_ERROR, error=exc
                    )
                    attempts.append(attempt)
                    last_error = exc
                    if "temperature" in str(exc).lower():
                        adaptations["removed_params"].append("temperature")
                    elif "max_tokens" in str(exc).lower() or "extra_body" in str(exc).lower():
                        adaptations["use_max_completion_tokens"] = True
                    else:
                        break

                except TransportError as exc:
                    lat_ms = (time.monotonic() - att_start_mono) * MS_PER_SECOND
                    attempt = self._record_attempt(
                        request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,
                        AttemptOutcome.TRANSPORT_ERROR, error=exc
                    )
                    attempts.append(attempt)
                    last_error = exc
                    # Fix #1: exponential back-off with full jitter (async-safe)
                    sleep_sec = _jitter_sleep_seconds(route_attempts)
                    logger.debug(
                        "TransportError on attempt %d/%d; sleeping %.0f ms before retry (Fix #1 jitter).",
                        attempt_count,
                        request.budget.max_physical_attempts,
                        sleep_sec * MS_PER_SECOND,
                    )
                    await asyncio.sleep(sleep_sec)

                except Exception as exc:
                    lat_ms = (time.monotonic() - att_start_mono) * MS_PER_SECOND
                    pe = ProviderError(GatewayErrorCode.PROVIDER_ERROR.value, str(exc))
                    attempt = self._record_attempt(
                        request, route, leg, attempt_count, att_kind, att_start_utc, lat_ms,

                        AttemptOutcome.PROVIDER_ERROR, error=pe
                    )
                    attempts.append(attempt)
                    last_error = pe
                    break

        return attempts, released_output, selected_route, final_val_summary, degradation, last_error

    # ── Public API ────────────────────────────────────────────────────────────

    def invoke_llm(self, request: LLMRequestV1) -> LLMResultV1:
        """Synchronous execution of LLMRequestV1 across provider route ladder.

        .. warning::
            Fix #2: This method is *blocking*.  It MUST NOT be called directly
            from an ``async`` function or an event-loop thread.  From async
            callers use ``await invoke_llm_async(request)`` or the helper
            ``await invoke_llm_in_executor(request)`` which offloads the
            blocking call to a thread-pool executor.
        """
        start_time_monotonic = time.monotonic()
        start_time_utc = time.time()
        deadline_ms = request.budget.deadline_ms

        initial_route, initial_leg = self._determine_initial_route(request)
        route_ladder = self._build_route_ladder(request, initial_route, initial_leg)

        attempts, released_output, selected_route, final_val_summary, degradation, last_error = (
            self._run_attempt_loop_sync(request, route_ladder, initial_route, initial_leg, start_time_monotonic)
        )

        return self._build_result(
            request, attempts, released_output, selected_route, final_val_summary,
            degradation, last_error, start_time_utc, start_time_monotonic, deadline_ms,
        )

    async def invoke_llm_async(self, request: LLMRequestV1) -> LLMResultV1:
        """Asynchronous execution of LLMRequestV1 across provider route ladder."""
        start_time_monotonic = time.monotonic()
        start_time_utc = time.time()
        deadline_ms = request.budget.deadline_ms

        initial_route, initial_leg = self._determine_initial_route(request)
        route_ladder = self._build_route_ladder(request, initial_route, initial_leg)

        attempts, released_output, selected_route, final_val_summary, degradation, last_error = (
            await self._run_attempt_loop_async(request, route_ladder, initial_route, initial_leg, start_time_monotonic)
        )

        return self._build_result(
            request, attempts, released_output, selected_route, final_val_summary,
            degradation, last_error, start_time_utc, start_time_monotonic, deadline_ms,
        )

    async def invoke_llm_in_executor(
        self,
        request: LLMRequestV1,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> LLMResultV1:
        """Run the synchronous engine in a thread-pool executor.

        Fix #2: Safe to call from within an async event loop.  Use this when
        you need the sync path (e.g., for a specific adapter that has no async
        implementation) but are running inside an async context.

        .. code-block:: python

            result = await engine.invoke_llm_in_executor(request)
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.invoke_llm, request)

    def _build_result(
        self,
        request: LLMRequestV1,
        attempts: List[PhysicalAttemptV1],
        released_output: Optional[OutputPayloadV1],
        selected_route: Optional[RouteV1],
        final_val_summary: ValidationSummaryV1,
        degradation: Optional[DegradationSummaryV1],
        last_error: Optional[LLMErrorV1],
        start_time_utc: float,
        start_time_monotonic: float,
        deadline_ms: int,
    ) -> LLMResultV1:
        """Assemble LLMResultV1 from attempt data (shared by sync and async paths)."""
        total_lat_ms = (time.monotonic() - start_time_monotonic) * MS_PER_SECOND
        tot_prompt = sum(a.prompt_tokens for a in attempts)
        tot_comp = sum(a.completion_tokens for a in attempts)
        tot_tokens = sum(a.total_tokens for a in attempts)
        tot_cost = sum(a.cost_usd for a in attempts)

        usage = UsageSummaryV1(
            prompt_tokens=tot_prompt,
            completion_tokens=tot_comp,
            total_tokens=tot_tokens,
            cost_usd=round(tot_cost, 6),
            pricing_snapshot_id=request.pricing_snapshot_id,
        )
        timing = TimingSummaryV1(
            started_at_utc=start_time_utc,
            completed_at_utc=time.time(),
            total_latency_ms=round(total_lat_ms, 2),
            deadline_ms=deadline_ms,
        )

        if released_output is not None:
            status = ResultStatus.DEGRADED if degradation else ResultStatus.SUCCEEDED
            err_dict = None
        else:
            status = ResultStatus.FAILED
            attempt_count = len(attempts)
            if attempt_count >= request.budget.max_physical_attempts:
                last_error = BudgetExhaustedError(
                    GatewayErrorCode.ATTEMPT_BUDGET_EXHAUSTED.value,
                    f"Max physical attempts ({request.budget.max_physical_attempts}) reached",
                )
            elif tot_cost >= request.budget.max_cost_usd:
                last_error = BudgetExhaustedError(
                    GatewayErrorCode.COST_BUDGET_EXHAUSTED.value, f"Max cost budget ${request.budget.max_cost_usd} exceeded"
                )
            elif total_lat_ms >= deadline_ms:
                last_error = BudgetExhaustedError(GatewayErrorCode.DEADLINE_EXHAUSTED.value, f"Request deadline of {deadline_ms}ms exceeded")

            err_dict = last_error.to_dict() if last_error else {
                "code": GatewayErrorCode.UNKNOWN_FAILURE.value,
                "error_class": ErrorClass.INTERNAL.value,
                "message": "Execution failed",
            }


        return LLMResultV1(
            request_id=request.request_id,
            policy_tag=request.policy_tag,
            status=status,
            output=released_output,
            selected_route=selected_route,
            validation=final_val_summary,
            attempts=attempts,
            usage=usage,
            timing=timing,
            degradation=degradation,
            error=err_dict,
            trace=request.trace_context,
        )


# ── Factory helpers (Fix #3) ─────────────────────────────────────────────────

def create_engine(registry: Optional[ProviderRegistry] = None) -> GatewayEngine:
    """Create a fresh, isolated GatewayEngine instance.

    Fix #3: Tests and DI containers should use this instead of mutating the
    global singleton.  Each call returns an independent engine with its own
    adapter map, so parallel test runs cannot interfere with each other.

    Example (pytest)::

        @pytest.fixture
        def engine() -> GatewayEngine:
            return create_engine()
    """
    return GatewayEngine(registry=registry)


def create_engine_for_registry(registry: ProviderRegistry) -> GatewayEngine:
    """Convenience alias for ``create_engine(registry=registry)``."""
    return GatewayEngine(registry=registry)


# ── Global engine singleton (backward compat) ────────────────────────────────
_GLOBAL_ENGINE = GatewayEngine()


def get_gateway_engine() -> GatewayEngine:
    return _GLOBAL_ENGINE


def invoke_llm(request: LLMRequestV1) -> LLMResultV1:
    """Public synchronous entry point for LLM Gateway.

    .. warning::
        Blocking call – do not use from async code.  Use ``invoke_llm_async``
        or ``invoke_llm_in_executor`` instead.
    """
    return _GLOBAL_ENGINE.invoke_llm(request)


async def invoke_llm_async(request: LLMRequestV1) -> LLMResultV1:
    """Public asynchronous entry point for LLM Gateway."""
    return await _GLOBAL_ENGINE.invoke_llm_async(request)


async def invoke_llm_in_executor(
    request: LLMRequestV1,
    executor: Optional[ThreadPoolExecutor] = None,
) -> LLMResultV1:
    """Run the global sync engine safely from within an async context.

    Fix #2: Wraps ``invoke_llm`` in ``loop.run_in_executor`` so it doesn't
    block the event loop.
    """
    return await _GLOBAL_ENGINE.invoke_llm_in_executor(request, executor=executor)
