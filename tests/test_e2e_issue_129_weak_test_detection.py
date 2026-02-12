"""
Tests for issue #129: pdd bug generates weak tests that pass without fixing the bug.

Covers:
- Step 8 verification detecting weak tests (no real assertions, parameter-only checks)
- Orchestrator hard stop logic on FAIL marker from Step 8
- Step 7 prompt anti-pattern compliance
- Integration scenario documenting the weak-test gap
"""

import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path

from pdd.agentic_bug_orchestrator import run_agentic_bug_orchestrator


# --- Fixtures ---


@pytest.fixture
def mock_dependencies(tmp_path):
    """Mock external dependencies for orchestrator tests."""
    mock_worktree_path = tmp_path / ".pdd" / "worktrees" / "fix-issue-1"
    mock_worktree_path.mkdir(parents=True, exist_ok=True)

    with patch("pdd.agentic_bug_orchestrator.run_agentic_task") as mock_run, \
         patch("pdd.agentic_bug_orchestrator.load_prompt_template") as mock_load, \
         patch("pdd.agentic_bug_orchestrator.console") as mock_console, \
         patch("pdd.agentic_bug_orchestrator._setup_worktree") as mock_worktree:

        mock_run.return_value = (True, "Step output", 0.1, "gpt-4")
        mock_load.return_value = "Prompt for {issue_number}"
        mock_worktree.return_value = (mock_worktree_path, None)

        yield mock_run, mock_load, mock_console


@pytest.fixture
def default_args(tmp_path):
    """Default arguments for the orchestrator."""
    return {
        "issue_url": "http://github.com/owner/repo/issues/1",
        "issue_content": "Bug description",
        "repo_owner": "owner",
        "repo_name": "repo",
        "issue_number": 1,
        "issue_author": "user",
        "issue_title": "Bug Title",
        "cwd": tmp_path,
        "verbose": False,
        "quiet": True,
    }


def _make_side_effect(step8_output, step7_output=None):
    """Helper: create a side_effect function with custom step 7/8 outputs."""
    def side_effect(*args, **kwargs):
        label = kwargs.get("label", "")
        if label == "step7":
            output = step7_output or "Generated test\nFILES_CREATED: test_file.py"
            return (True, output, 0.1, "gpt-4")
        if label == "step8":
            return (True, step8_output, 0.1, "gpt-4")
        return (True, f"Output for {label}", 0.1, "gpt-4")
    return side_effect


# --- Scenario 1: Step 8 Verification Detects Weak Tests ---


class TestStep8DetectsWeakTests:
    """Step 8 should reject tests with no meaningful assertions."""

    def test_assert_true_only_triggers_fail(self, mock_dependencies, default_args):
        """A test containing only `assert True` should trigger FAIL marker from Step 8."""
        mock_run, _, _ = mock_dependencies
        mock_run.side_effect = _make_side_effect(
            step8_output="FAIL: Test does not work as expected - test only contains trivial assertions"
        )

        success, msg, _, _, _ = run_agentic_bug_orchestrator(**default_args)

        assert success is False
        assert "Stopped at Step 8" in msg
        assert "verification failed" in msg.lower()

    def test_parameter_existence_check_triggers_fail(self, mock_dependencies, default_args):
        """A test that only checks parameter existence (not behavior) should trigger FAIL."""
        mock_run, _, _ = mock_dependencies
        mock_run.side_effect = _make_side_effect(
            step8_output="FAIL: Test does not work as expected - test checks parameter acceptance, not behavior"
        )

        success, msg, _, _, _ = run_agentic_bug_orchestrator(**default_args)

        assert success is False
        assert "Stopped at Step 8" in msg

    def test_no_assert_statements_triggers_fail(self, mock_dependencies, default_args):
        """A test with no assert statements at all should trigger FAIL."""
        mock_run, _, _ = mock_dependencies
        mock_run.side_effect = _make_side_effect(
            step8_output="FAIL: Test does not work as expected - no assertions found in test"
        )

        success, msg, _, _, _ = run_agentic_bug_orchestrator(**default_args)

        assert success is False
        assert "Stopped at Step 8" in msg


# --- Scenario 2: Step 8 Detects Tests That Pass on Buggy Code ---


class TestStep8DetectsTestsPassingOnBuggyCode:
    """Tests that pass on buggy code (exit 0) should be rejected."""

    def test_test_passes_on_buggy_code_triggers_fail(self, mock_dependencies, default_args):
        """If agent reports test passes on current buggy code, Step 8 should FAIL."""
        mock_run, _, _ = mock_dependencies
        mock_run.side_effect = _make_side_effect(
            step8_output="FAIL: Test does not work as expected - test passes on current buggy code"
        )

        success, msg, _, _, _ = run_agentic_bug_orchestrator(**default_args)

        assert success is False
        assert "Stopped at Step 8" in msg

    def test_test_fails_for_wrong_reason_triggers_fail(self, mock_dependencies, default_args):
        """If test fails with ImportError/SyntaxError (not the bug), Step 8 should FAIL."""
        mock_run, _, _ = mock_dependencies
        mock_run.side_effect = _make_side_effect(
            step8_output="FAIL: Test does not work as expected - test fails with ImportError, not related to bug"
        )

        success, msg, _, _, _ = run_agentic_bug_orchestrator(**default_args)

        assert success is False
        assert "Stopped at Step 8" in msg

    def test_test_correctly_detects_bug_passes(self, mock_dependencies, default_args):
        """If test correctly fails on buggy code for the right reason, workflow continues."""
        mock_run, _, _ = mock_dependencies
        mock_run.side_effect = _make_side_effect(
            step8_output="PASS: Test correctly detects the bug - assertion on output behavior fails as expected"
        )

        success, msg, _, _, _ = run_agentic_bug_orchestrator(**default_args)

        assert success is True
        assert "Investigation complete" in msg


# --- Scenario 3: Orchestrator Step 8 Hard Stop Logic ---


class TestStep8HardStopLogic:
    """Verify the orchestrator correctly handles FAIL/PASS markers."""

    def test_fail_marker_stops_workflow(self, mock_dependencies, default_args):
        """Exact FAIL marker string causes orchestrator to return (False, ...)."""
        mock_run, _, _ = mock_dependencies
        mock_run.side_effect = _make_side_effect(
            step8_output="FAIL: Test does not work as expected"
        )

        success, msg, cost, model, files = run_agentic_bug_orchestrator(**default_args)

        assert success is False
        assert "Stopped at Step 8" in msg
        # Steps 1-8 should have run (9 calls including step 5.5)
        assert mock_run.call_count == 9

    def test_pass_marker_continues_to_step_9(self, mock_dependencies, default_args):
        """PASS marker allows workflow to continue past Step 8."""
        mock_run, _, _ = mock_dependencies
        mock_run.side_effect = _make_side_effect(
            step8_output="PASS: Test correctly detects the bug"
        )

        success, msg, _, _, _ = run_agentic_bug_orchestrator(**default_args)

        assert success is True
        # All 11 steps should run
        assert mock_run.call_count == 11

    def test_no_marker_ambiguous_continues(self, mock_dependencies, default_args):
        """Step 8 output with no FAIL/PASS marker is treated as success (documenting gap)."""
        mock_run, _, _ = mock_dependencies
        mock_run.side_effect = _make_side_effect(
            step8_output="The test looks reasonable and should catch the bug."
        )

        success, msg, _, _, _ = run_agentic_bug_orchestrator(**default_args)

        # Current behavior: no FAIL marker means workflow continues
        assert success is True
        assert mock_run.call_count == 11


# --- Scenario 4: Step 7 Prompt Anti-Pattern Compliance ---


class TestStep7PromptCompliance:
    """Verify Step 7 prompt includes anti-pattern warnings."""

    def test_step7_prompt_contains_caller_behavior_section(self):
        """Step 7 prompt template should contain the critical anti-pattern warning section."""
        from pdd.load_prompt_template import load_prompt_template
        template = load_prompt_template("agentic_bug_step7_generate_LLM")
        assert template is not None, "Step 7 prompt template not found"
        assert "Critical: Testing Caller Behavior Bugs" in template
        assert "DO NOT: Test that the callee rejects wrong parameters" in template

    def test_step7_receives_prior_step_context(self, mock_dependencies, default_args):
        """Step 7 receives accumulated context from steps 1-6."""
        mock_run, mock_load, _ = mock_dependencies

        # Track what context keys are available when step 7 template is formatted
        captured_contexts = []

        original_return = "Prompt for {issue_number}"

        def load_side_effect(name):
            # For step 7, use a template that references prior steps
            if "step7" in name:
                return "Step7 prompt: {step1_output} {step6_output} {issue_number}"
            return original_return

        mock_load.side_effect = load_side_effect

        def run_side_effect(*args, **kwargs):
            label = kwargs.get("label", "")
            if label == "step7":
                return (True, "Generated test\nFILES_CREATED: test_file.py", 0.1, "gpt-4")
            return (True, f"Output for {label}", 0.1, "gpt-4")

        mock_run.side_effect = run_side_effect

        success, msg, _, _, _ = run_agentic_bug_orchestrator(**default_args)

        # If the orchestrator formatted step7 template without KeyError,
        # it means prior step outputs were available in context
        assert success is True


# --- Scenario 5: Integration - Weak Test Slips Through (Documenting the Gap) ---


class TestWeakTestGapDocumentation:
    """
    Simulate the #486 quiet-flag scenario: Step 7 produces a weak test,
    Step 8 fails to catch it, workflow succeeds — proving the gap exists.
    This test establishes a regression baseline for future fixes.
    """

    def test_weak_test_passes_through_when_step8_misses_it(self, mock_dependencies, default_args):
        """
        When Step 8 doesn't detect a weak test (no FAIL marker),
        the workflow succeeds even though the test is insufficient.
        This documents the current gap from issue #129.
        """
        mock_run, _, _ = mock_dependencies

        def side_effect(*args, **kwargs):
            label = kwargs.get("label", "")
            if label == "step7":
                # Step 7 produces a weak test that only checks parameter acceptance
                return (True, (
                    "def test_quiet_flag():\n"
                    "    # Only checks that function accepts quiet parameter\n"
                    "    result = generate(prompt='test', quiet=True)\n"
                    "    assert result is not None\n"
                    "\nFILES_CREATED: test_quiet.py"
                ), 0.1, "gpt-4")
            if label == "step8":
                # Step 8 FAILS TO DETECT the weak test — no FAIL marker emitted
                return (True, (
                    "The test checks that the quiet flag is accepted by the generate function. "
                    "This should work to verify the fix."
                ), 0.1, "gpt-4")
            return (True, f"Output for {label}", 0.1, "gpt-4")

        mock_run.side_effect = side_effect

        success, msg, _, _, _ = run_agentic_bug_orchestrator(**default_args)

        # The workflow succeeds despite the weak test — this IS the bug from #129
        assert success is True, (
            "Expected workflow to succeed (documenting the gap): "
            "Step 8 missed the weak test, so no FAIL marker was emitted"
        )
        assert mock_run.call_count == 11
