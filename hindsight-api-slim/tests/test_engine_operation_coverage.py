"""Automated architectural coverage test for MemoryEngine.

Asserts that every public per-bank engine method (methods accepting ``bank_id`` and ``request_context``)
either invokes ``_validate_operation`` / ``_consume_preauthorized`` or is explicitly registered on the allowed
list of non-operational/internal engine routines. This prevents regressions like #3312 and #3831.
"""

import inspect
import pytest

from hindsight_api.engine.memory_engine import MemoryEngine


# Allowlist of public methods taking bank_id & request_context that are explicitly non-operational
# or internal orchestration wrappers.
EXEMPT_ENGINE_METHODS = {
    # Bank life-cycle management / internal existence checks
    "create_bank",
    "get_bank",
    "list_banks",
    "delete_bank",
    "get_bank_stats",
    "update_bank_config",
    "get_bank_config",
    "reset_bank_config",
    "compute_mental_model_is_stale",
    # Audit log / Telemetry reporting
    "list_audit_logs",
    "audit_log_stats",
    "list_llm_requests",
    "llm_request_stats",
    # Dry-run / Internal file retrieval
    "extract_dry_run",
    "retrieve_bank_file",
    "delete_memory_unit",
    "refresh_mental_model",
    # Async task queuing wrappers (validation occurs worker-side or via delegate)
    "retain_async",
    "submit_async_file_retain",
    "import_documents_async",
    "export_documents_async",
    "submit_export_documents_async",
}


def test_all_public_per_bank_methods_emit_operation_validation():
    """Assert every public per-bank method invokes _validate_operation or _consume_preauthorized."""
    public_methods = [
        (name, method)
        for name, method in inspect.getmembers(MemoryEngine, predicate=inspect.iscoroutinefunction)
        if not name.startswith("_")
    ]

    unvalidated_methods = []

    for name, method in public_methods:
        if name in EXEMPT_ENGINE_METHODS:
            continue

        try:
            sig = inspect.signature(method)
        except (ValueError, TypeError):
            continue

        params = sig.parameters
        if "bank_id" in params and "request_context" in params:
            try:
                source = inspect.getsource(method)
            except OSError:
                continue

            has_validator = (
                "_validate_operation" in source
                or "_consume_preauthorized" in source
                or "_operation_validator" in source
            )

            if not has_validator:
                unvalidated_methods.append(name)

    assert not unvalidated_methods, (
        f"The following per-bank MemoryEngine methods accept `bank_id` and `request_context` "
        f"but do not invoke `_validate_operation` or `_operation_validator`: {unvalidated_methods}. "
        f"Add operation validation or add to EXEMPT_ENGINE_METHODS if explicitly non-operational."
    )
