"""Structured Output Validation and Semantic Validator Registry (v1).

Fix #1 – JSON Schema validation replaced with full ``jsonschema`` Draft-7
implementation.  The old hand-rolled checker only covered top-level type,
required keys and additional-properties; it missed nested objects, field
types, enums, patterns and ``$ref``.  ``jsonschema`` handles all of those.

Fix #1b – Markdown fenced-block stripping is now case-insensitive so that
```JSON ... ``` (uppercase) is stripped as well as ```json ... ```.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm_gateway.constants import SemanticValidatorId

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import jsonschema.  If it is absent we fall back to the original
# shallow checker and emit a single WARNING at import time so operators notice.
# ---------------------------------------------------------------------------
try:
    import jsonschema
    from jsonschema import ValidationError as _JSValidationError
    _JSONSCHEMA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JSONSCHEMA_AVAILABLE = False
    logger.warning(
        "llm_gateway.validators: 'jsonschema' package not found. "
        "JSON Schema validation will fall back to a shallow structural check. "
        "Install it with: pip install jsonschema>=4.21.0"
    )

# Type for semantic validator functions: (parsed_json, validator_inputs) -> (is_valid, errors)
SemanticValidatorFn = Callable[[Any, Dict[str, Any]], Tuple[bool, List[str]]]

# Regex to strip ```json ... ``` or ``` ... ``` blocks, case-insensitive
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?(.*?)```$", re.DOTALL)


class ValidatorRegistry:
    """Registry of deterministic, call-site-specific semantic validators."""

    def __init__(self) -> None:
        self._validators: Dict[str, SemanticValidatorFn] = {}
        self._register_default_validators()

    def register_validator(self, validator_id: str, fn: SemanticValidatorFn) -> None:
        self._validators[validator_id] = fn

    def get_validator(self, validator_id: str) -> Optional[SemanticValidatorFn]:
        return self._validators.get(validator_id)

    def _register_default_validators(self) -> None:
        # 1. No-op validator for free-text or unrestricted output
        def none_validator(output: Any, inputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
            return True, []

        self.register_validator(SemanticValidatorId.NONE.value, none_validator)

        # 2. Gather action semantic validator (for generic domain gather proposals)
        def gather_action_validator(output: Any, inputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
            if not isinstance(output, dict):
                return False, ["Output must be a JSON object"]
            allowed_tools = inputs.get("allowed_tools", [])
            tool_name = output.get("tool")
            if tool_name is not None and tool_name not in allowed_tools:
                return False, [f"Proposed tool '{tool_name}' is not in allowed tools: {allowed_tools}"]
            return True, []

        self.register_validator(SemanticValidatorId.GATHER_ACTION.value, gather_action_validator)

        # 3. Confirmation payload validator
        def confirmation_payload_validator(output: Any, inputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
            if not isinstance(output, dict):
                return False, ["Confirmation payload must be a JSON object"]
            expected_domain = inputs.get("domain")
            if expected_domain and output.get("domain") and output.get("domain") != expected_domain:
                return False, [f"Domain mismatch: expected {expected_domain}, got {output.get('domain')}"]
            return True, []

        self.register_validator(SemanticValidatorId.CONFIRMATION_PAYLOAD.value, confirmation_payload_validator)



_GLOBAL_VALIDATOR_REGISTRY = ValidatorRegistry()


def get_validator_registry() -> ValidatorRegistry:
    return _GLOBAL_VALIDATOR_REGISTRY


# ---------------------------------------------------------------------------
# JSON syntax parsing
# ---------------------------------------------------------------------------

def validate_json_syntax(raw_text: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    """Attempt to parse raw text into JSON.

    Strips Markdown fenced code blocks (case-insensitive) before parsing.
    """
    if not raw_text or not raw_text.strip():
        return False, None, "Empty response cannot be parsed as JSON"

    cleaned = raw_text.strip()

    # Strip ```json ... ``` or ```JSON ... ``` or plain ``` ... ```  (case-insensitive)
    fence_match = _FENCE_RE.match(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        return True, parsed, None
    except json.JSONDecodeError as exc:
        return False, None, f"JSON decode error: {exc.msg} at line {exc.lineno} col {exc.colno}"


# ---------------------------------------------------------------------------
# JSON Schema validation
# ---------------------------------------------------------------------------

def validate_json_schema(
    parsed_json: Any,
    json_schema: Optional[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """Full JSON Schema compliance check (Draft-7 via ``jsonschema``).

    If ``jsonschema`` is not installed, falls back to the previous shallow
    structural check (top-level type + required keys + additionalProperties).

    Fix #1: Replaced hand-rolled checker with real Draft-7 validation so that
    nested objects, field types, enum values, patterns, and ``$ref`` are all
    validated correctly.
    """
    if not json_schema:
        return True, []

    if _JSONSCHEMA_AVAILABLE:
        return _validate_with_jsonschema(parsed_json, json_schema)
    else:
        return _validate_shallow(parsed_json, json_schema)


def _validate_with_jsonschema(
    parsed_json: Any,
    json_schema: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """Validate using the ``jsonschema`` library (full Draft-7)."""
    errors: List[str] = []
    try:
        validator_cls = jsonschema.Draft7Validator
        validator = validator_cls(json_schema)
        for error in sorted(validator.iter_errors(parsed_json), key=str):
            path = " -> ".join(str(p) for p in error.absolute_path) if error.absolute_path else "root"
            errors.append(f"[{path}] {error.message}")
    except _JSValidationError as exc:
        errors.append(str(exc.message))
    except Exception as exc:
        # Schema itself is malformed; treat as a validation failure
        errors.append(f"JSON Schema error: {exc}")

    return len(errors) == 0, errors


def _validate_shallow(
    parsed_json: Any,
    json_schema: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """Fallback shallow validator (used when jsonschema is not installed)."""
    errors: List[str] = []
    expected_type = json_schema.get("type")
    if expected_type == "object" and not isinstance(parsed_json, dict):
        errors.append(f"Expected object, got {type(parsed_json).__name__}")
        return False, errors
    elif expected_type == "array" and not isinstance(parsed_json, list):
        errors.append(f"Expected array, got {type(parsed_json).__name__}")
        return False, errors

    if isinstance(parsed_json, dict):
        required_keys = json_schema.get("required", [])
        for key in required_keys:
            if key not in parsed_json:
                errors.append(f"Missing required field '{key}'")

        properties = json_schema.get("properties", {})
        additional_allowed = json_schema.get("additionalProperties", True)
        if additional_allowed is False:
            for key in parsed_json:
                if key not in properties:
                    errors.append(f"Forbidden additional property '{key}'")

    return len(errors) == 0, errors
