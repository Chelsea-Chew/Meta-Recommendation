"""
Probe and Test Script for Anthropic Claude via WP5 LLM Gateway.

Usage:
    python scripts/probe_anthropic.py [--model MODEL_NAME] [--dry-run] [--async] [--test-fallback]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

# Ensure backend root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(SCRIPT_DIR)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from dotenv import find_dotenv, load_dotenv

# Load environment variables
load_dotenv(find_dotenv())

from llm_gateway import (
    AnthropicAdapter,
    AttemptOutcome,
    BudgetConfigV1,
    GenerationConfigV1,
    LLMRequestV1,
    MessageV1,
    PolicyContextV1,
    PolicyLeg,
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
    get_gateway_engine,
    get_provider_registry,
    invoke_llm,
    invoke_llm_async,
)
from llm_gateway.adapters.fake import FakeLLMAdapter, ScriptedResponse


def check_anthropic_credentials():
    print("=" * 65)
    print("  LLM GATEWAY: ANTHROPIC CLAUDE CREDENTIAL & REGISTRY STATUS")
    print("=" * 65)

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").strip()
    version = os.getenv("ANTHROPIC_VERSION", "2023-06-01").strip()
    has_key = bool(api_key and not api_key.startswith("Some "))

    print(f"• Anthropic Claude Service:")
    print(f"    - ANTHROPIC_API_KEY: {'[CONFIGURED]' if has_key else '[MISSING / PLACEHOLDER]'}")
    print(f"    - Base URL:          {base_url}")
    print(f"    - API Version:       {version}")

    registry = get_provider_registry()
    anthropic_prov = registry.get_provider("anthropic")
    if anthropic_prov:
        print("\n• Registered Claude Models in ProviderCapabilityV1:")
        for m in anthropic_prov.models:
            print(f"    - {m.model_id:30s} context_window={m.context_window:<7d} max_output={m.max_output_tokens}")
    else:
        print("\n[!] 'anthropic' provider not found in registry!")

    pricing = registry.get_pricing_snapshot("default-pricing-v1")
    print("\n• Frozen Pricing Snapshot (default-pricing-v1):")
    for m in (anthropic_prov.models if anthropic_prov else []):
        rate = pricing.get_price("anthropic", m.model_id)
        print(f"    - {m.model_id:30s} ${rate.input_price_per_1m:6.2f} / 1M in,  ${rate.output_price_per_1m:6.2f} / 1M out")

    print("=" * 65 + "\n")
    return has_key


def build_test_request(
    model_id: str,
    prompt: str = "Explain in one sentence why multi-provider LLM gateways increase reliability.",
    fallback_route: RouteV1 | None = None,
) -> LLMRequestV1:
    routing = RoutingPolicyV1(
        mode=RoutingMode.BASELINE,
        baseline_route=RouteV1(provider_id="anthropic", model_id=model_id, policy_leg=PolicyLeg.BASELINE),
        fallback_routes=[fallback_route] if fallback_route else [],
    )

    return LLMRequestV1(
        request_id=f"probe-anthropic-{int(time.time())}",
        policy_tag="conversation.test_probe",
        messages=[
            MessageV1(role=RequestRole.SYSTEM, content="You are a concise software systems architect."),
            MessageV1(role=RequestRole.USER, content=prompt),
        ],
        generation=GenerationConfigV1(temperature=0.3, max_output_tokens=100),
        response_contract=ResponseContractV1(kind=ResponseKind.TEXT),
        routing_policy=routing,
        budget=BudgetConfigV1(deadline_ms=15000, max_physical_attempts=2, max_cost_usd=0.25),
        policy_context=PolicyContextV1(contains_itinerary_state=False, request_mode=RequestMode.SINGLE_DOMAIN),
    )


def run_probe(model_id: str, is_dry_run: bool = False, test_async: bool = False, test_fallback: bool = False):
    engine = get_gateway_engine()

    if is_dry_run:
        print("[INFO] Running in DRY-RUN mode using simulated FakeLLMAdapter...")
        fake_anthropic = FakeLLMAdapter()
        if test_fallback:
            # Simulate Anthropic 529 outage, recovering to fake Azure/OpenAI
            fake_anthropic.enqueue_response(
                ScriptedResponse(
                    error_to_raise=Exception("Claude server overloaded (HTTP 529)"),
                )
            )
            fake_fallback = FakeLLMAdapter()
            fake_fallback.enqueue_response(
                ScriptedResponse(
                    text="Fallback route succeeded: Multi-provider failover preserves availability during outages.",
                    prompt_tokens=45,
                    completion_tokens=18,
                )
            )
            engine.register_adapter("anthropic", fake_anthropic)
            engine.register_adapter("openai", fake_fallback)
        else:
            fake_anthropic.enqueue_response(
                ScriptedResponse(
                    text="Anthropic Claude dry-run: Gateway eliminates single-point-of-failure risks.",
                    prompt_tokens=38,
                    completion_tokens=14,
                )
            )
            engine.register_adapter("anthropic", fake_anthropic)

    fallback_route = None
    if test_fallback:
        fallback_route = RouteV1(provider_id="openai", model_id="gpt-4o-mini", policy_leg=PolicyLeg.FALLBACK)

    req = build_test_request(model_id=model_id, fallback_route=fallback_route)

    mode_label = "ASYNC" if test_async else "SYNC"
    print(f"--- Executing {mode_label} Probe on 'anthropic:{model_id}' ---")
    t0 = time.time()

    if test_async:
        result = asyncio.run(invoke_llm_async(req))
    else:
        result = invoke_llm(req)

    elapsed = (time.time() - t0) * 1000.0

    print(f"\n[Result Overview]")
    print(f"  • Status:          {result.status.value.upper()}")
    print(f"  • Selected Route:  {result.selected_route.provider_id if result.selected_route else 'None'}:{result.selected_route.model_id if result.selected_route else 'None'} ({result.selected_route.policy_leg.value if result.selected_route else 'None'})")
    print(f"  • Attempts Made:   {len(result.attempts)}")
    print(f"  • Total Tokens:    {result.usage.total_tokens} (Prompt: {result.usage.prompt_tokens}, Completion: {result.usage.completion_tokens})")
    print(f"  • Total Cost:      ${result.usage.cost_usd:.6f}")
    print(f"  • Total Latency:   {elapsed:.1f}ms (Engine recorded: {result.timing.total_latency_ms:.1f}ms)")

    if result.degradation:
        print(f"  • Degradation:     {result.degradation.reason_code} (from {result.degradation.from_route.provider_id} to {result.degradation.to_route.provider_id})")

    print(f"\n[Attempt Breakdown]")
    for i, att in enumerate(result.attempts, 1):
        print(f"  Attempt #{i}:")
        print(f"    - Provider:  {att.provider_id}:{att.model_id}")
        print(f"    - Leg:       {att.policy_leg.value}")
        print(f"    - Kind:      {att.attempt_kind.value}")
        print(f"    - Outcome:   {att.outcome.value}")
        print(f"    - Latency:   {att.latency_ms:.1f}ms")
        print(f"    - Cost:      ${att.cost_usd:.6f}")
        if att.error_code:
            print(f"    - Error:     [{att.error_code}] {att.error_message}")

    print(f"\n[Released Output]")
    if result.output and result.output.text:
        print(f"  \"{result.output.text.strip()}\"")
    elif result.error:
        print(f"  [ERROR] {result.error.code}: {result.error.message}")

    print("-" * 65 + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="Probe Anthropic Claude via WP5 LLM Gateway.")
    parser.add_argument("--model", default="claude-3-5-haiku-20241022", help="Claude model ID to test")
    parser.add_argument("--dry-run", action="store_true", help="Run without live network calls using fake adapter")
    parser.add_argument("--async", dest="test_async", action="store_true", help="Test async execution path")
    parser.add_argument("--test-fallback", action="store_true", help="Simulate outage and fallback ladder")

    args = parser.parse_args()

    has_key = check_anthropic_credentials()

    dry_run = args.dry_run
    if not has_key and not dry_run:
        print("[NOTICE] No live ANTHROPIC_API_KEY detected. Automatically running in --dry-run mode.\n")
        dry_run = True

    run_probe(
        model_id=args.model,
        is_dry_run=dry_run,
        test_async=args.test_async,
        test_fallback=args.test_fallback,
    )


if __name__ == "__main__":
    main()
