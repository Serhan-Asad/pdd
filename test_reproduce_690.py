"""
Reproduce GitHub Issue #690:
When `pdd sync` test step fails, the entire sync is marked Failed
even though code generation succeeded. There is no mechanism to
accept the generated code and move on.

Bug location: pdd/sync_orchestration.py lines 2462-2465
    if not success:
        if not errors:
            errors.append(f"Operation '{operation}' failed.")
        break

The while loop immediately breaks on any operation failure, marks
the entire sync as failed (success = `not errors`), and discards
all prior progress. A successful generate followed by a failing
test results in the same top-level outcome as a failed generate.
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


# ---------------------------------------------------------------------------
# Simulate the fail-fast logic from sync_worker_logic (lines 1624-2488)
# ---------------------------------------------------------------------------

@dataclass
class MockDecision:
    """Mirrors SyncDecision from sync_determine_operation."""
    operation: str
    reason: str
    estimated_cost: float = 0.0
    confidence: float = 1.0
    details: Optional[Dict[str, Any]] = None


def simulated_sync_worker_logic(
    operation_sequence: List[str],
    operation_results: Dict[str, bool],
    budget: float = 10.0,
) -> Dict[str, Any]:
    """
    A minimal reproduction of the while-loop logic from
    sync_orchestration.sync_worker_logic().

    This mirrors the actual code structure:
      - calls sync_determine_operation() in a loop
      - dispatches to operation handlers
      - on failure: appends error, breaks immediately
      - returns {'success': not errors, 'operations_completed': [...], ...}

    Parameters
    ----------
    operation_sequence : list of str
        Operations that sync_determine_operation would return in order.
        Must end with 'all_synced' or include a failing step.
    operation_results : dict
        Mapping of operation name -> bool indicating success/failure.
    budget : float
        Simulated budget.
    """
    operations_completed: List[str] = []
    errors: List[str] = []
    current_cost = 0.0
    call_index = 0

    # ---- The while True loop (mirrors lines 1624-2465) ----
    while True:
        # Budget check (mirrors lines 1625-1632)
        if current_cost >= budget:
            errors.append(f"Budget of ${budget:.2f} exceeded.")
            break

        # Get next operation (mirrors lines 1640-1651)
        if call_index >= len(operation_sequence):
            # Safety: shouldn't happen in well-formed test
            errors.append("Ran out of operations (unexpected).")
            break
        operation = operation_sequence[call_index]
        call_index += 1

        # Terminal operations (mirrors lines 1812-1827)
        if operation in ['all_synced', 'nothing']:
            break
        if operation in ['fail_and_request_manual_merge', 'error']:
            errors.append(f"Terminal error: {operation}")
            break

        # ---- Execute operation (mirrors lines 1881-1882) ----
        success = False

        # Simulate operation execution (mirrors lines 1891-2321)
        success = operation_results.get(operation, False)
        cost = 0.01  # nominal
        current_cost += cost

        # ---- Result handling (mirrors lines 2359-2368) ----
        if success:
            operations_completed.append(operation)

        # ---- The bug: fail-fast break (mirrors lines 2462-2465) ----
        if not success:
            if not errors:
                errors.append(f"Operation '{operation}' failed.")
            break

    # ---- Return dict (mirrors lines 2478-2488) ----
    return {
        'success': not errors,
        'operations_completed': operations_completed,
        'errors': errors,
        'error': "; ".join(errors) if errors else None,
        'total_cost': current_cost,
    }


# ===========================================================================
# Tests
# ===========================================================================

class TestIssue690FailFastOnTestFailure:
    """
    Reproduce the scenario: generate succeeds, then test fails.
    The entire sync is reported as Failed with no way to keep the
    generated code.
    """

    def test_generate_succeeds_but_test_fails_marks_whole_sync_failed(self):
        """
        When generate succeeds and test fails, the sync result has
        success=False even though usable code was generated.
        """
        result = simulated_sync_worker_logic(
            operation_sequence=['generate', 'test', 'all_synced'],
            operation_results={'generate': True, 'test': False},
        )

        # BUG: The entire sync is marked as failed
        assert result['success'] is False

        # The error message references 'test', not 'generate'
        assert len(result['errors']) == 1
        assert "Operation 'test' failed." in result['errors'][0]

        # generate completed successfully and IS recorded
        assert 'generate' in result['operations_completed']

        # test did NOT complete (it failed), so it is NOT in operations_completed
        assert 'test' not in result['operations_completed']

    def test_no_mechanism_to_accept_generated_code(self):
        """
        The return dict has no field indicating that partial progress
        (e.g. code generation) can be accepted despite test failure.
        """
        result = simulated_sync_worker_logic(
            operation_sequence=['generate', 'test', 'all_synced'],
            operation_results={'generate': True, 'test': False},
        )

        # There is no 'partial_success' or 'accept_partial' field
        assert 'partial_success' not in result
        assert 'accept_partial' not in result

        # The only success indicator is the top-level 'success' which is False
        assert result['success'] is False

        # Even though operations_completed shows generate worked,
        # calling code only checks result['success'] and treats
        # the entire sync as a failure.
        assert result['operations_completed'] == ['generate']

    def test_generate_failure_looks_identical_to_test_failure(self):
        """
        A sync where generate itself fails produces the same top-level
        success=False as one where generate succeeds but test fails.
        The caller cannot distinguish partial success from total failure.
        """
        # Scenario A: generate fails immediately
        result_gen_fail = simulated_sync_worker_logic(
            operation_sequence=['generate', 'test', 'all_synced'],
            operation_results={'generate': False, 'test': True},
        )

        # Scenario B: generate succeeds, test fails
        result_test_fail = simulated_sync_worker_logic(
            operation_sequence=['generate', 'test', 'all_synced'],
            operation_results={'generate': True, 'test': False},
        )

        # Both have the exact same top-level success value
        assert result_gen_fail['success'] is False
        assert result_test_fail['success'] is False

        # But the underlying situations are very different:
        # - gen_fail: no usable code was produced
        # - test_fail: code WAS produced and could be accepted
        assert result_gen_fail['operations_completed'] == []
        assert result_test_fail['operations_completed'] == ['generate']

        # The bug: there is no way for the caller to differentiate these
        # two outcomes using the 'success' field alone.

    def test_break_prevents_subsequent_operations(self):
        """
        The break on test failure prevents any subsequent operations
        (like 'fix') from running, even though the pipeline could
        potentially recover.
        """
        result = simulated_sync_worker_logic(
            operation_sequence=['generate', 'test', 'fix', 'test', 'all_synced'],
            operation_results={
                'generate': True,
                'test': False,
                'fix': True,
            },
        )

        # The loop broke after 'test' failed — 'fix' never ran
        assert result['success'] is False
        assert 'fix' not in result['operations_completed']
        assert result['operations_completed'] == ['generate']


class TestIssue690VerifyAgainstRealCode:
    """
    Verify that the actual sync_orchestration.py contains the
    fail-fast pattern described in the issue.
    """

    def test_break_on_failure_pattern_exists_in_source(self):
        """
        Confirm the bug pattern exists in the actual source code:
            if not success:
                if not errors:
                    errors.append(f"Operation '{operation}' failed.")
                break
        """
        import inspect
        from pathlib import Path

        source_path = Path(__file__).parent / "pdd" / "sync_orchestration.py"
        source = source_path.read_text(encoding="utf-8")

        # The fail-fast pattern: unconditional break on any operation failure
        assert "if not success:" in source
        assert "errors.append(f\"Operation '{operation}' failed.\")" in source

        # Find the break that follows the error append.
        # The break is indented at the same level as the 'if not errors:' block,
        # meaning it executes unconditionally when success is False.
        lines = source.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "if not success:":
                # Look ahead for the break within the next few lines
                remaining = lines[i:i+5]
                has_break = any(l.strip() == "break" for l in remaining)
                if has_break:
                    # Found the pattern — the break is unconditional
                    # (not inside an else or additional condition)
                    break
        else:
            pytest.fail(
                "Could not find the 'if not success: ... break' pattern "
                "in sync_orchestration.py"
            )

    def test_success_derived_from_errors_list(self):
        """
        Confirm that the return value's 'success' is derived solely
        from whether the errors list is non-empty, with no consideration
        of operations_completed.
        """
        from pathlib import Path

        source_path = Path(__file__).parent / "pdd" / "sync_orchestration.py"
        source = source_path.read_text(encoding="utf-8")

        # The result dict uses `not errors` as the success value
        assert "'success': not errors," in source

        # There is no 'partial_success' key in the return dict
        assert "'partial_success'" not in source


class TestIssue690MultiStepPipeline:
    """
    Test longer pipeline scenarios to show the fail-fast behavior
    at different stages.
    """

    def test_crash_failure_after_generate_and_test(self):
        """
        generate -> test -> crash (fails): entire sync fails,
        even though code and tests were successfully generated.
        """
        result = simulated_sync_worker_logic(
            operation_sequence=['generate', 'test', 'crash', 'all_synced'],
            operation_results={
                'generate': True,
                'test': True,
                'crash': False,
            },
        )

        assert result['success'] is False
        assert result['operations_completed'] == ['generate', 'test']
        assert "Operation 'crash' failed." in result['errors'][0]

    def test_full_pipeline_succeeds(self):
        """
        Sanity check: when all operations succeed, sync reports success.
        """
        result = simulated_sync_worker_logic(
            operation_sequence=['generate', 'test', 'crash', 'verify', 'all_synced'],
            operation_results={
                'generate': True,
                'test': True,
                'crash': True,
                'verify': True,
            },
        )

        assert result['success'] is True
        assert result['errors'] == []
        assert result['operations_completed'] == ['generate', 'test', 'crash', 'verify']
