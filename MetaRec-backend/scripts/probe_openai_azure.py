"""
Probe and Test Script for OpenAI and Azure OpenAI via WP5 LLM Gateway.

Usage:
    python scripts/probe_openai_azure.py [--provider openai|azure_openai|all] [--model MODEL_NAME] [--dry-run]
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

from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

from llm_gateway import (
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
    get_gateway_engine,
    get_provider_registry,
    invoke_llm,
    invoke_llm_async,
)
from llm_gateway.adapters.fake import FakeLLMAdapter, ScriptedResponse
from llm_usage import UsageLedger, push_ledger, reset_ledger


def check_credentials():
    print("=" * 60)
    print("  LLM GATEWAY: CREDENTIAL & PROVIDER STATUS")
    print("=" * 60)

    # 1. OpenAI Direct
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    has_openai = bool(openai_key and not openai_key.startswith("Some "))

    print(f"• OpenAI Direct:")
    print(f"    - OPENAI_API_KEY: {'[CONFIGURED]' if has_openai else '[MISSING / PLACEHOLDER]'}")
    print(f"    - Base URL:       {openai_base}")

    # 2. Azure OpenAI
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    azure_key = (os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")).strip()
    azure_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview").strip()
    has_azure = bool(
        azure_endpoint
        and not azure_endpoint.startswith("Some ")
        and azure_key
        and not azure_key.startswith("Some ")
    )

    print(f"• Azure OpenAI:")
    print(f"    - AZURE_OPENAI_ENDPOINT:    {azure_endpoint or '[MISSING]'}")
    print(f"    - API Key:                  {'[CONFIGURED]' if has_azure else '[MISSING / PLACEHOLDER]'}")
    print(f"    - AZURE_OPENAI_API_VERSION: {azure_version}")

    # 3. Registry Check
    registry = get_provider_registry()
    print("\n• Registered Provider Capabilities:")
    for prov in registry.list_providers():
        models = ", ".join(m.model_id for m in prov.models)
        print(f"    - [{prov.provider_id}] adapter={prov.adapter_kind}, models=({models})")

    print("=" * 60 + "\n")
    return has_openai, has_azure


def build_test_request(
    provider_id: str,
    model_id: str,
    prompt: str = "Explain in one sentence why multi-provider AI gateways improve reliability.",
    fallback_route: RouteV1 | None = None,
) -> LLMRequestV1:
    routing = RoutingPolicyV1(
        mode=RoutingMode.BASELINE,
        baseline_route=RouteV1(provider_id=provider_id, model_id=model_id, policy_leg=PolicyLeg.BASELINE),
        fallback_routes=[fallback_route] if fallback_route else [],
    )

    return LLMRequestV1(
        request_id=f"probe-{provider_id}-{int(time.time())}",
        policy_tag="conversation.test_probe",
        messages=[
            MessageV1(role=RequestRole.SYSTEM, content="You are a helpful software architecture assistant."),
            MessageV1(role=RequestRole.USER, content=prompt),
        ],
        generation=GenerationConfigV1(temperature=0.3, max_output_tokens=100),
        response_contract=ResponseContractV1(kind=ResponseKind.TEXT),
        routing_policy=routing,
        budget=BudgetConfigV1(deadline_ms=15000, max_physical_attempts=2, max_cost_usd=0.20),
        policy_context=PolicyContextV1(contains_itinerary_state=False, request_mode=RequestMode.SINGLE_DOMAIN),
    )


def run_probe(provider_id: str, model_id: str, is_dry_run: bool = False, test_async: bool = False):
    print(f"\n[PROBE] Testing Provider: '{provider_id}' | Model: '{model_id}' | Mode: {'DRY RUN' if is_dry_run else 'LIVE'}")
    print("-" * 60)

    ledger = UsageLedger()
    token = push_ledger(ledger)

    try:
        req = build_test_request(provider_id, model_id)

        if is_dry_run:
            engine = get_gateway_engine()
            fake_adapter = FakeLLMAdapter()
            fake_adapter.enqueue_response(
                ScriptedResponse(
                    text=f"[DRY-RUN SIMULATION] Response from mock {provider_id}:{model_id}",
                    prompt_tokens=28,
                    completion_tokens=14,
                )
            )
            engine.register_adapter(provider_id, fake_adapter)

        start_t = time.perf_counter()
        if test_async:
            print("Invoking asynchronously via invoke_llm_async()...")
            result = asyncio.run(invoke_llm_async(req))
        else:
            print("Invoking synchronously via invoke_llm()...")
            result = invoke_llm(req)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        print(f"Status:          {result.status.value.upper()}")
        print(f"Execution Time:  {elapsed_ms:.2f} ms")
        print(f"Attempts Made:   {len(result.attempts)}")

        for idx, att in enumerate(result.attempts, 1):
            print(f"  Attempt {idx}: Provider={att.provider_id}, Model={att.model_id}, Outcome={att.outcome.value}, Latency={att.latency_ms:.1f}ms")

        if result.status == ResultStatus.SUCCEEDED:
            print(f"\nResponse Text:")
            print(f"  \"{result.output.text.strip()}\"")
            print(f"\nToken & Cost Accounting:")
            print(f"  Prompt Tokens:     {result.usage.prompt_tokens}")
            print(f"  Completion Tokens: {result.usage.completion_tokens}")
            print(f"  Total Tokens:      {result.usage.total_tokens}")
            print(f"  Estimated Cost:    ${result.usage.cost_usd:.6f} USD")
            print(f"  Ledger Events:     {len(ledger.events)} event(s) recorded")
            print(">>> TEST PASSED <<<")
            return True
        else:
            print(f"\nExecution FAILED:")
            print(f"  Error Class:   {result.error.get('error_class') if result.error else 'Unknown'}")
            print(f"  Error Code:    {result.error.get('error_code') if result.error else 'Unknown'}")
            print(f"  Message:       {result.error.get('message') if result.error else 'Unknown'}")
            print(">>> TEST FAILED <<<")
            return False

    except Exception as exc:
        print(f"\nUNHANDLED EXCEPTION: {exc}")
        return False
    finally:
        reset_ledger(token)


def run_fallback_test(primary_prov: str, primary_model: str, fallback_prov: str, fallback_model: str):
    print(f"\n[FALLBACK TEST] Primary: '{primary_prov}' -> Fallback: '{fallback_prov}'")
    print("-" * 60)
    engine = get_gateway_engine()

    # Create scripted fake adapters to demonstrate automatic failover
    fake_primary = FakeLLMAdapter()
    fake_primary.enqueue_response(
        ScriptedResponse(
            error_to_raise=Exception("Simulated primary outage 503 Server Unavailable")
        )
    )
    fake_fallback = FakeLLMAdapter()
    fake_fallback.enqueue_response(
        ScriptedResponse(
            text="[RECOVERED] Successfully routed to fallback provider after primary outage.",
            prompt_tokens=35,
            completion_tokens=15,
        )
    )

    engine.register_adapter(primary_prov, fake_primary)
    engine.register_adapter(fallback_prov, fake_fallback)

    fb_route = RouteV1(provider_id=fallback_prov, model_id=fallback_model, policy_leg=PolicyLeg.FALLBACK)
    req = build_test_request(primary_prov, primary_model, fallback_route=fb_route)

    res = invoke_llm(req)
    print(f"Result Status:   {res.status.value.upper()}")
    print(f"Selected Route:  Provider={res.selected_route.provider_id}, Model={res.selected_route.model_id}, Leg={res.selected_route.policy_leg.value}")
    print(f"Attempts Count:  {len(res.attempts)}")
    for idx, a in enumerate(res.attempts, 1):
        print(f"  Attempt {idx}: Provider={a.provider_id}, Outcome={a.outcome.value}")
    if res.status in (ResultStatus.SUCCEEDED, ResultStatus.DEGRADED):
        print(f"Output: \"{res.output.text}\"")
        print(">>> FALLBACK RECOVERY VERIFIED <<<")
    else:
        print(">>> FALLBACK TEST FAILED <<<")


def main():
    parser = argparse.ArgumentParser(description="Test LLM Gateway with OpenAI, OpenAI Base, and Azure OpenAI")
    parser.add_argument("--provider", choices=["openai", "openai_base", "azure_openai", "all", "fallback"], default="all")
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--azure-model", default="gpt-4o-mini", help="Azure OpenAI deployment name (default: gpt-4o-mini)")
    parser.add_argument("--dry-run", action="store_true", help="Run without calling external network (using mock adapter)")
    parser.add_argument("--async-test", action="store_true", help="Test async invocation path")
    args = parser.parse_args()

    has_openai, has_azure = check_credentials()

    if args.dry_run:
        print("[INFO] Running in DRY-RUN mode. External network calls will be mocked.")
    else:
        if args.provider in ("openai", "openai_base", "all") and not has_openai:
            print("[NOTICE] OpenAI credentials not configured in .env. Will run in dry-run mode.")
        if args.provider in ("azure_openai", "all") and not has_azure:
            print("[NOTICE] Azure credentials not configured in .env. Will run Azure test in dry-run mode.")

    # 1. Test OpenAI
    if args.provider in ("openai", "all"):
        is_mock = args.dry_run or not has_openai
        run_probe("openai", args.openai_model, is_dry_run=is_mock, test_async=args.async_test)

    # 2. Test OpenAI Base (Direct HTTP)
    if args.provider in ("openai_base", "all"):
        is_mock = args.dry_run or not has_openai
        run_probe("openai_base", args.openai_model, is_dry_run=is_mock, test_async=args.async_test)

    # 3. Test Azure OpenAI
    if args.provider in ("azure_openai", "all"):
        is_mock = args.dry_run or not has_azure
        run_probe("azure_openai", args.azure_model, is_dry_run=is_mock, test_async=args.async_test)

    # 4. Test Fallback
    if args.provider in ("fallback", "all"):
        run_fallback_test("openai", args.openai_model, "azure_openai", args.azure_model)


if __name__ == "__main__":
    main()
