"""
E2E tests for Issue #503: Agentic commands should reject closed GitHub issues.

These tests exercise the full CLI-to-orchestrator code path using Click's
CliRunner, which simulates real user invocation of `pdd bug`, `pdd fix`,
and `pdd change`. Only the GitHub API layer and LLM orchestrator are mocked;
the entry-point functions (agentic_bug.py, agentic_e2e_fix.py, agentic_change.py)
run their REAL code, including URL parsing, issue data inspection, and the
(currently missing) state check.

This differs from the unit tests in test_agentic_closed_issue.py, which mock
_fetch_issue_data and call the entry-point functions directly. These E2E tests
verify the bug from the user's perspective: invoking a CLI command with a
closed-issue URL.

All 3 primary tests FAIL on the current (buggy) code because:
- The CLI commands call their respective run_agentic_* functions
- Those functions fetch issue data (mocked to return state=closed)
- No state check exists, so the orchestrator is invoked anyway

They should PASS once the state check is added to each entry point.

Issue: https://github.com/gltanaka/pdd/issues/503
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch
from click.testing import CliRunner

# --- Shared test data ---

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

# Standard orchestrator return value (5-tuple)
ORCH_RETURN = (True, "Orchestrator completed", 1.5, "gpt-4o", ["file.py"])


class TestE2EIssue503ClosedIssueGuard:
    """
    E2E tests verifying that CLI commands reject closed GitHub issues
    before launching the expensive LLM orchestrator.

    These tests use Click's CliRunner to invoke the real CLI commands,
    exercising the full code path from CLI argument parsing through
    the entry-point function's state check (or lack thereof).
    """

    def test_pdd_bug_cli_rejects_closed_issue(self):
        """
        E2E: `pdd bug <closed-issue-url>` should fail with a message about
        the issue being closed, and should NOT invoke the orchestrator.

        Current buggy behavior: The CLI proceeds through the full workflow,
        invokes the orchestrator, and returns success.

        Expected behavior after fix: The CLI prints a "closed" message and
        exits without calling the orchestrator.
        """
        from pdd.cli import cli

        runner = CliRunner()

        with patch("pdd.agentic_bug._check_gh_cli", return_value=True), \
             patch("pdd.agentic_bug._fetch_issue_data",
                   return_value=(CLOSED_ISSUE_DATA, None)), \
             patch("pdd.agentic_bug._fetch_comments", return_value=""), \
             patch("pdd.agentic_bug._ensure_repo_context", return_value=True), \
             patch("pdd.agentic_bug.run_agentic_bug_orchestrator",
                   return_value=ORCH_RETURN) as mock_orchestrator, \
             patch("pdd.agentic_bug.console"):

            result = runner.invoke(
                cli,
                ["--quiet", "--no-core-dump", "bug", CLOSED_ISSUE_URL],
                catch_exceptions=False,
            )

        # On buggy code: orchestrator IS called → this assertion fails
        # On fixed code: orchestrator NOT called → passes
        assert mock_orchestrator.call_count == 0, (
            f"BUG (Issue #503): The bug orchestrator was called {mock_orchestrator.call_count} "
            f"time(s) for a closed issue. Expected pdd bug to reject closed issues "
            f"before launching the LLM workflow.\nCLI output:\n{result.output}"
        )

    def test_pdd_fix_cli_rejects_closed_issue(self):
        """
        E2E: `pdd fix <closed-issue-url>` should fail with a message about
        the issue being closed, and should NOT invoke the orchestrator.

        Current buggy behavior: The CLI proceeds through the full workflow.
        Expected behavior after fix: Early exit with "closed" message.
        """
        from pdd.cli import cli

        runner = CliRunner()

        with patch("pdd.agentic_e2e_fix._check_gh_cli", return_value=True), \
             patch("pdd.agentic_e2e_fix._fetch_issue_data",
                   return_value=(CLOSED_ISSUE_DATA, None)), \
             patch("pdd.agentic_e2e_fix._fetch_issue_comments", return_value=""), \
             patch("pdd.agentic_e2e_fix._find_working_directory",
                   return_value=(Path("/tmp/pdd_test_repo"), None, False)), \
             patch("pdd.agentic_e2e_fix.run_agentic_e2e_fix_orchestrator",
                   return_value=ORCH_RETURN) as mock_orchestrator, \
             patch("pdd.agentic_e2e_fix.console"):

            result = runner.invoke(
                cli,
                ["--quiet", "--no-core-dump", "fix", CLOSED_ISSUE_URL],
                catch_exceptions=False,
            )

        assert mock_orchestrator.call_count == 0, (
            f"BUG (Issue #503): The fix orchestrator was called {mock_orchestrator.call_count} "
            f"time(s) for a closed issue. Expected pdd fix to reject closed issues "
            f"before launching the LLM workflow.\nCLI output:\n{result.output}"
        )

    def test_pdd_change_cli_rejects_closed_issue(self):
        """
        E2E: `pdd change <closed-issue-url>` should fail with a message about
        the issue being closed, and should NOT invoke the orchestrator.

        Current buggy behavior: The CLI proceeds through the full workflow.
        Expected behavior after fix: Early exit with "closed" message.
        """
        from pdd.cli import cli

        runner = CliRunner()

        with patch("pdd.agentic_change._check_gh_cli", return_value=True), \
             patch("pdd.agentic_change._run_gh_command") as mock_gh_cmd, \
             patch("pdd.agentic_change._setup_repository",
                   return_value=Path("/tmp/pdd_test_repo")), \
             patch("pdd.agentic_change.run_agentic_change_orchestrator",
                   return_value=ORCH_RETURN) as mock_orchestrator, \
             patch("pdd.agentic_change.console"):

            # First call: fetch issue JSON; Second call: fetch comments
            mock_gh_cmd.side_effect = [
                (True, json.dumps(CLOSED_ISSUE_DATA)),   # issue fetch
                (True, "[]"),                             # comments fetch
            ]

            result = runner.invoke(
                cli,
                ["--quiet", "--no-core-dump", "change", CLOSED_ISSUE_URL],
                catch_exceptions=False,
            )

        assert mock_orchestrator.call_count == 0, (
            f"BUG (Issue #503): The change orchestrator was called {mock_orchestrator.call_count} "
            f"time(s) for a closed issue. Expected pdd change to reject closed issues "
            f"before launching the LLM workflow.\nCLI output:\n{result.output}"
        )

    def test_pdd_fix_cli_force_bypasses_closed_issue_guard(self):
        """
        E2E: `pdd fix --force <closed-issue-url>` should bypass the
        closed-issue guard and proceed to the orchestrator.

        This test verifies that --force (which already exists on the fix
        command for branch mismatch override) also overrides the new
        closed-issue check.

        NOTE: On the current buggy code this test PASSES because no guard
        exists and the orchestrator is always called. It serves as a
        specification for the intended post-fix behavior.
        """
        from pdd.cli import cli

        runner = CliRunner()

        with patch("pdd.agentic_e2e_fix._check_gh_cli", return_value=True), \
             patch("pdd.agentic_e2e_fix._fetch_issue_data",
                   return_value=(CLOSED_ISSUE_DATA, None)), \
             patch("pdd.agentic_e2e_fix._fetch_issue_comments", return_value=""), \
             patch("pdd.agentic_e2e_fix._find_working_directory",
                   return_value=(Path("/tmp/pdd_test_repo"), None, False)), \
             patch("pdd.agentic_e2e_fix.run_agentic_e2e_fix_orchestrator",
                   return_value=ORCH_RETURN) as mock_orchestrator, \
             patch("pdd.agentic_e2e_fix.console"):

            result = runner.invoke(
                cli,
                ["--quiet", "--no-core-dump", "fix", "--force", CLOSED_ISSUE_URL],
                catch_exceptions=False,
            )

        # With --force, the orchestrator MUST be called even for closed issues
        assert mock_orchestrator.call_count == 1, (
            f"--force should bypass the closed-issue guard. The orchestrator was called "
            f"{mock_orchestrator.call_count} time(s), expected 1.\nCLI output:\n{result.output}"
        )

    def test_pdd_bug_cli_proceeds_on_open_issue(self):
        """
        E2E regression test: `pdd bug <open-issue-url>` should proceed
        normally and invoke the orchestrator.

        This ensures the closed-issue fix doesn't accidentally block
        all issues.
        """
        from pdd.cli import cli

        runner = CliRunner()

        with patch("pdd.agentic_bug._check_gh_cli", return_value=True), \
             patch("pdd.agentic_bug._fetch_issue_data",
                   return_value=(OPEN_ISSUE_DATA, None)), \
             patch("pdd.agentic_bug._fetch_comments", return_value=""), \
             patch("pdd.agentic_bug._ensure_repo_context", return_value=True), \
             patch("pdd.agentic_bug.run_agentic_bug_orchestrator",
                   return_value=ORCH_RETURN) as mock_orchestrator, \
             patch("pdd.agentic_bug.console"):

            result = runner.invoke(
                cli,
                ["--quiet", "--no-core-dump", "bug", OPEN_ISSUE_URL],
                catch_exceptions=False,
            )

        # Open issues must ALWAYS reach the orchestrator
        assert mock_orchestrator.call_count == 1, (
            f"Open issues must proceed to the orchestrator. The orchestrator was called "
            f"{mock_orchestrator.call_count} time(s), expected 1.\nCLI output:\n{result.output}"
        )

    def test_pdd_bug_cli_closed_issue_output_mentions_closed(self):
        """
        E2E: When `pdd bug` rejects a closed issue, the CLI output should
        contain the word "closed" so the user understands why the command
        failed.

        On buggy code: output contains "Success: True" (no rejection).
        On fixed code: output contains "closed" (clear rejection message).
        """
        from pdd.cli import cli

        runner = CliRunner()

        with patch("pdd.agentic_bug._check_gh_cli", return_value=True), \
             patch("pdd.agentic_bug._fetch_issue_data",
                   return_value=(CLOSED_ISSUE_DATA, None)), \
             patch("pdd.agentic_bug._fetch_comments", return_value=""), \
             patch("pdd.agentic_bug._ensure_repo_context", return_value=True), \
             patch("pdd.agentic_bug.run_agentic_bug_orchestrator",
                   return_value=ORCH_RETURN) as mock_orchestrator, \
             patch("pdd.agentic_bug.console"):

            result = runner.invoke(
                cli,
                ["--quiet", "--no-core-dump", "bug", CLOSED_ISSUE_URL],
                catch_exceptions=False,
            )

        # The orchestrator should NOT have been called
        assert mock_orchestrator.call_count == 0, (
            f"BUG (Issue #503): The bug orchestrator was called for a closed issue."
        )

        # The CLI output should mention that the issue is closed
        output_lower = result.output.lower()
        assert "closed" in output_lower, (
            f"BUG (Issue #503): CLI output should mention 'closed' when "
            f"rejecting a closed issue, but got:\n{result.output}"
        )
