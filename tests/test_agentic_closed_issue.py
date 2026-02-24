"""
Tests that agentic commands (bug, fix, change, test) reject closed GitHub issues
without proceeding to the expensive LLM workflow.

These tests FAIL on the current (buggy) code — all four entry points fetch issue
data that includes "state": "closed" but never inspect that field, so the
downstream orchestrator is called anyway.

They should PASS once the state check is added to each entry point.

Issue: https://github.com/gltanaka/pdd/issues/503
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# --- Shared fixture data ---

CLOSED_ISSUE_DATA = {
    "state": "closed",
    "title": "better llm selector",
    "body": "Please add a better llm selector.",
    "user": {"login": "gltanaka"},
    "comments_url": "https://api.github.com/repos/gltanaka/pdd/issues/1/comments",
    "number": 1,
}

OPEN_ISSUE_DATA = {
    "state": "open",
    "title": "some open issue",
    "body": "This is an open issue that should be processed normally.",
    "user": {"login": "gltanaka"},
    "comments_url": "https://api.github.com/repos/gltanaka/pdd/issues/99/comments",
    "number": 99,
}

CLOSED_ISSUE_URL = "https://github.com/gltanaka/pdd/issues/1"
OPEN_ISSUE_URL = "https://github.com/gltanaka/pdd/issues/99"

# Standard orchestrator return value (5-tuple) so mocks don't cause ValueError
ORCH_RETURN = (True, "Orchestrator completed", 1.5, "gpt-4o", ["file.py"])


# =============================================================================
# Test 1: run_agentic_bug rejects closed issue (primary bug)
# =============================================================================

def test_agentic_bug_rejects_closed_issue():
    """
    run_agentic_bug should return (False, <msg>, ...) without calling the
    orchestrator when the GitHub issue state is "closed".

    On the CURRENT (buggy) code this test FAILS because:
      - _fetch_issue_data returns {"state": "closed", ...}
      - agentic_bug.py never inspects the "state" field
      - run_agentic_bug_orchestrator is called anyway
      - mock_orchestrator.assert_not_called() raises AssertionError
    """
    with patch("pdd.agentic_bug._check_gh_cli", return_value=True), \
         patch("pdd.agentic_bug._fetch_issue_data", return_value=(CLOSED_ISSUE_DATA, None)), \
         patch("pdd.agentic_bug._fetch_comments", return_value=""), \
         patch("pdd.agentic_bug._ensure_repo_context", return_value=True), \
         patch("pdd.agentic_bug.run_agentic_bug_orchestrator",
               return_value=ORCH_RETURN) as mock_orchestrator, \
         patch("pdd.agentic_bug.console"):

        from pdd.agentic_bug import run_agentic_bug
        success, msg, cost, model, changed_files = run_agentic_bug(
            CLOSED_ISSUE_URL, quiet=True
        )

    # This assertion fails on buggy code: orchestrator IS called for closed issue
    mock_orchestrator.assert_not_called()


# =============================================================================
# Test 2: run_agentic_change rejects closed issue (primary bug)
# =============================================================================

def test_agentic_change_rejects_closed_issue():
    """
    run_agentic_change should return (False, <msg>, ...) without calling the
    orchestrator when the GitHub issue state is "closed".

    On the CURRENT (buggy) code this test FAILS because:
      - _run_gh_command returns the closed issue JSON
      - agentic_change.py never inspects the "state" field
      - run_agentic_change_orchestrator is called anyway
    """
    with patch("pdd.agentic_change._check_gh_cli", return_value=True), \
         patch("pdd.agentic_change._run_gh_command") as mock_gh_cmd, \
         patch("pdd.agentic_change._setup_repository", return_value=Path("/tmp/pdd_test_repo")), \
         patch("pdd.agentic_change.run_agentic_change_orchestrator",
               return_value=ORCH_RETURN) as mock_orchestrator, \
         patch("pdd.agentic_change.console"):

        # First call fetches the issue; second call fetches comments
        mock_gh_cmd.side_effect = [
            (True, json.dumps(CLOSED_ISSUE_DATA)),  # issue fetch
            (True, "[]"),                            # comments fetch
        ]

        from pdd.agentic_change import run_agentic_change
        success, msg, cost, model, changed_files = run_agentic_change(
            CLOSED_ISSUE_URL, quiet=True
        )

    # This assertion fails on buggy code: orchestrator IS called for closed issue
    mock_orchestrator.assert_not_called()


# =============================================================================
# Test 3: run_agentic_e2e_fix rejects closed issue (primary bug)
# =============================================================================

def test_agentic_e2e_fix_rejects_closed_issue():
    """
    run_agentic_e2e_fix should return (False, <msg>, ...) without calling the
    orchestrator when the GitHub issue state is "closed".

    On the CURRENT (buggy) code this test FAILS because:
      - _fetch_issue_data returns {"state": "closed", ...}
      - agentic_e2e_fix.py never inspects the "state" field before the
        _find_working_directory call
      - run_agentic_e2e_fix_orchestrator is called anyway
    """
    with patch("pdd.agentic_e2e_fix._check_gh_cli", return_value=True), \
         patch("pdd.agentic_e2e_fix._fetch_issue_data", return_value=(CLOSED_ISSUE_DATA, None)), \
         patch("pdd.agentic_e2e_fix._fetch_issue_comments", return_value=""), \
         patch("pdd.agentic_e2e_fix._find_working_directory",
               return_value=(Path("/tmp/pdd_test_repo"), None, False)), \
         patch("pdd.agentic_e2e_fix.run_agentic_e2e_fix_orchestrator",
               return_value=ORCH_RETURN) as mock_orchestrator, \
         patch("pdd.agentic_e2e_fix.console"):

        from pdd.agentic_e2e_fix import run_agentic_e2e_fix
        success, msg, cost, model, changed_files = run_agentic_e2e_fix(
            CLOSED_ISSUE_URL, quiet=True
        )

    # This assertion fails on buggy code: orchestrator IS called for closed issue
    mock_orchestrator.assert_not_called()


# =============================================================================
# Test 4: run_agentic_test rejects closed issue (4th affected file)
# =============================================================================

def test_agentic_test_rejects_closed_issue():
    """
    run_agentic_test should return (False, <msg>, ...) without calling the
    orchestrator when the GitHub issue state is "closed".

    On the CURRENT (buggy) code this test FAILS because:
      - _fetch_issue_data returns {"state": "closed", ...}
      - agentic_test.py extracts "state" only as LLM metadata but never
        uses it as a guard to prevent the workflow
      - run_agentic_test_orchestrator is called anyway
    """
    with patch("pdd.agentic_test._check_gh_cli", return_value=True), \
         patch("pdd.agentic_test._fetch_issue_data", return_value=(CLOSED_ISSUE_DATA, None)), \
         patch("pdd.agentic_test._ensure_repo_context", return_value=(True, "/tmp/pdd_test_repo")), \
         patch("pdd.agentic_test.run_agentic_test_orchestrator",
               return_value=ORCH_RETURN) as mock_orchestrator, \
         patch("pdd.agentic_test.console"):

        from pdd.agentic_test import run_agentic_test
        success, msg, cost, model, changed_files = run_agentic_test(
            CLOSED_ISSUE_URL, quiet=True
        )

    # This assertion fails on buggy code: orchestrator IS called for closed issue
    mock_orchestrator.assert_not_called()


# =============================================================================
# Test 5: Open issues still proceed normally (regression prevention)
# =============================================================================

def test_agentic_bug_proceeds_on_open_issue():
    """
    run_agentic_bug should NOT block open issues — the state check must only
    reject closed ones. This test guards against overly broad fixes that would
    accidentally reject all issues.
    """
    with patch("pdd.agentic_bug._check_gh_cli", return_value=True), \
         patch("pdd.agentic_bug._fetch_issue_data", return_value=(OPEN_ISSUE_DATA, None)), \
         patch("pdd.agentic_bug._fetch_comments", return_value=""), \
         patch("pdd.agentic_bug._ensure_repo_context", return_value=True), \
         patch("pdd.agentic_bug.run_agentic_bug_orchestrator",
               return_value=ORCH_RETURN) as mock_orchestrator, \
         patch("pdd.agentic_bug.console"):

        from pdd.agentic_bug import run_agentic_bug
        success, msg, cost, model, changed_files = run_agentic_bug(
            OPEN_ISSUE_URL, quiet=True
        )

    # The orchestrator MUST have been called for an open issue
    mock_orchestrator.assert_called_once()
    assert success is True


# =============================================================================
# Test 6: force=True bypasses the closed-issue guard (edge case)
# =============================================================================

def test_agentic_e2e_fix_force_overrides_closed_issue_check():
    """
    When force=True is passed to run_agentic_e2e_fix, the closed-issue guard
    should be bypassed and the orchestrator should still be invoked.

    NOTE: This test will PASS on the current (buggy) code because the check
    doesn't exist yet and the orchestrator is always called. It serves as a
    specification for the intended post-fix behavior and will prevent
    regressions if --force support is added.
    """
    with patch("pdd.agentic_e2e_fix._check_gh_cli", return_value=True), \
         patch("pdd.agentic_e2e_fix._fetch_issue_data", return_value=(CLOSED_ISSUE_DATA, None)), \
         patch("pdd.agentic_e2e_fix._fetch_issue_comments", return_value=""), \
         patch("pdd.agentic_e2e_fix._find_working_directory",
               return_value=(Path("/tmp/pdd_test_repo"), None, False)), \
         patch("pdd.agentic_e2e_fix.run_agentic_e2e_fix_orchestrator",
               return_value=ORCH_RETURN) as mock_orchestrator, \
         patch("pdd.agentic_e2e_fix.console"):

        from pdd.agentic_e2e_fix import run_agentic_e2e_fix
        success, msg, cost, model, changed_files = run_agentic_e2e_fix(
            CLOSED_ISSUE_URL, force=True, quiet=True
        )

    # With force=True the orchestrator must be called even for a closed issue
    mock_orchestrator.assert_called_once()


# =============================================================================
# Test 7: Closed-issue message is actionable (mentions issue number and force)
# =============================================================================

def test_closed_issue_message_is_actionable():
    """
    The error message returned when rejecting a closed issue should be
    actionable: it should mention the issue number and reference 'force'
    so users know how to override.

    On the CURRENT (buggy) code this test FAILS because:
      - No state check exists, so the function never returns a closed-issue message
      - The function proceeds to the orchestrator instead of returning early
    """
    closed_issue_42 = {
        "state": "closed",
        "title": "old bug",
        "body": "Done.",
        "user": {"login": "author"},
        "comments_url": "",
        "number": 42,
    }

    with patch("pdd.agentic_bug._check_gh_cli", return_value=True), \
         patch("pdd.agentic_bug._fetch_issue_data", return_value=(closed_issue_42, None)), \
         patch("pdd.agentic_bug._fetch_comments", return_value=""), \
         patch("pdd.agentic_bug._ensure_repo_context", return_value=True), \
         patch("pdd.agentic_bug.run_agentic_bug_orchestrator",
               return_value=ORCH_RETURN) as mock_orchestrator, \
         patch("pdd.agentic_bug.console"):

        from pdd.agentic_bug import run_agentic_bug
        success, msg, cost, model, changed_files = run_agentic_bug(
            "https://github.com/owner/repo/issues/42",
            quiet=True
        )

    # Verify the function rejected the closed issue (not called orchestrator)
    mock_orchestrator.assert_not_called()
    # Verify the message is actionable
    assert success is False, "Should fail for closed issue"
    assert "42" in msg, f"Message should contain the issue number 42, got: {msg}"
    assert "force" in msg.lower(), f"Message should mention 'force' for override, got: {msg}"
