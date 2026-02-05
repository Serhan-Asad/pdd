"""
E2E Test for Issue #469: pdd sessions cleanup shows misleading success message
when all cleanups fail

This E2E test uses Click's CliRunner to invoke the actual CLI command, exercising
the full code path from command invocation through output and exit code handling.

Bug: When running `pdd sessions cleanup --all --force` and all cleanup operations
fail, the command displays a misleading success message with a green checkmark (✓)
and exits with code 0 (success).

Root Cause: In `pdd/commands/sessions.py:282-284`:
1. Line 282 unconditionally prints "✓ Successfully cleaned up 0 session(s)"
2. No `sys.exit(1)` call when failures occur
3. Missing `import sys`

E2E Test Strategy:
- Use CliRunner to invoke the actual CLI command (real user path)
- Mock only the external dependencies (RemoteSessionManager, CloudConfig)
- Exercise the real sessions.cleanup() function without mocking it
- Verify exit codes and output messages match expected behavior

The test should:
- FAIL on the current buggy code (misleading message shown, exit code 0)
- PASS once the bug is fixed (no success message with 0 count, exit code 1)

Issue: https://github.com/promptdriven/pdd/issues/469
"""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from pdd.commands.sessions import sessions


# --- Mock Data ---

@dataclass
class MockSessionInfo:
    """Mock session info for testing."""
    session_id: str
    project_name: str
    cloud_url: str
    status: str
    last_heartbeat: str
    created_at: str = "2024-01-01T00:00:00Z"
    project_path: str = "/test/path"
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@pytest.mark.e2e
class TestIssue469SessionsCleanupE2E:
    """
    E2E tests using CliRunner to verify the sessions cleanup exit code bug.

    These tests exercise the full CLI command path including:
    - Command parsing and validation
    - Async execution
    - Output formatting
    - Exit code handling

    Unlike unit tests, we mock only external dependencies (API calls),
    not the command logic itself.
    """

    @pytest.fixture
    def runner(self):
        """Fixture to provide a CliRunner for testing Click commands."""
        return CliRunner()

    @pytest.fixture
    def mock_sessions(self):
        """Fixture providing sample session data."""
        return [
            MockSessionInfo(
                session_id="4d540d05-1234-5678-9abc-def012345671",
                project_name="test-project",
                cloud_url="https://pdd.dev/connect/4d540d05",
                status="active",
                last_heartbeat="2024-01-01T10:00:00Z",
            ),
            MockSessionInfo(
                session_id="5e651e16-2345-6789-abcd-ef1234567892",
                project_name="another-project",
                cloud_url="https://pdd.dev/connect/5e651e16",
                status="active",
                last_heartbeat="2024-01-01T09:00:00Z",
            ),
        ]

    def test_cleanup_all_fail_wrong_exit_code(self, runner, mock_sessions):
        """
        E2E Test: `pdd sessions cleanup --all --force` when all cleanups fail.

        User scenario:
        1. User has 2 active remote sessions
        2. User runs `pdd sessions cleanup --all --force`
        3. All cleanup operations fail (e.g., network errors, server issues)

        CURRENT BUGGY BEHAVIOR:
        - Exit code is 0 (wrong - should be 1 to indicate failure)

        EXPECTED BEHAVIOR (after fix):
        - Exit code should be 1 to indicate failure

        This test FAILS on buggy code (exit code 0), PASSES after fix (exit code 1).
        """
        # Mock CloudConfig to return a valid JWT
        with patch("pdd.commands.sessions.CloudConfig") as mock_cloud_config:
            mock_cloud_config.get_jwt_token.return_value = "mock-jwt-token"

            # Mock RemoteSessionManager class
            with patch("pdd.commands.sessions.RemoteSessionManager") as mock_manager_class:
                # Mock list_sessions as a class/static method
                mock_manager_class.list_sessions = AsyncMock(return_value=mock_sessions)

                # Create mock instances for cleanup that always fail (raise exception)
                async def failing_deregister():
                    raise Exception("Simulated network error")

                def create_failing_instance(*args, **kwargs):
                    instance = MagicMock()
                    instance.deregister = AsyncMock(side_effect=failing_deregister)
                    return instance

                mock_manager_class.side_effect = create_failing_instance

                # Run the actual CLI command
                result = runner.invoke(sessions, ["cleanup", "--all", "--force"])

                # THE BUG: Exit code should be 1 when all cleanups fail
                # This assertion FAILS on buggy code (exit code is 0), PASSES after fix
                assert result.exit_code == 1, (
                    f"BUG DETECTED (Issue #469): Exit code is {result.exit_code} but should be 1 "
                    f"when all cleanup operations fail.\n\n"
                    f"This prevents automation/scripts from detecting failures.\n\n"
                    f"Output:\n{result.output}"
                )

    def test_cleanup_all_fail_shows_misleading_success_message(self, runner, mock_sessions):
        """
        E2E Test: `pdd sessions cleanup --all --force` shows misleading success message.

        User scenario:
        1. User has 2 active remote sessions
        2. User runs `pdd sessions cleanup --all --force`
        3. All cleanup operations fail

        CURRENT BUGGY BEHAVIOR:
        - Output shows "✓ Successfully cleaned up 0 session(s)" (misleading!)
        - This is confusing because a green checkmark implies success

        EXPECTED BEHAVIOR (after fix):
        - Output should NOT show "Successfully cleaned up 0 session(s)"

        This test FAILS on buggy code (message IS shown), PASSES after fix (message NOT shown).
        """
        with patch("pdd.commands.sessions.CloudConfig") as mock_cloud_config:
            mock_cloud_config.get_jwt_token.return_value = "mock-jwt-token"

            with patch("pdd.commands.sessions.RemoteSessionManager") as mock_manager_class:
                # Mock list_sessions as a class/static method
                mock_manager_class.list_sessions = AsyncMock(return_value=mock_sessions)

                # Create mock instances for cleanup that always fail
                async def failing_deregister():
                    raise Exception("Simulated network error")

                def create_failing_instance(*args, **kwargs):
                    instance = MagicMock()
                    instance.deregister = AsyncMock(side_effect=failing_deregister)
                    return instance

                mock_manager_class.side_effect = create_failing_instance

                result = runner.invoke(sessions, ["cleanup", "--all", "--force"])

                # THE BUG: Should NOT show "Successfully cleaned up 0 session(s)"
                # This assertion FAILS on buggy code (message IS shown), PASSES after fix
                assert "Successfully cleaned up 0 session(s)" not in result.output, (
                    f"BUG DETECTED (Issue #469): The misleading success message "
                    f"'Successfully cleaned up 0 session(s)' was displayed even though "
                    f"all cleanup operations failed.\n\n"
                    f"This is confusing to users - a green checkmark (✓) implies success, "
                    f"but cleaning up 0 sessions is not a success when operations failed.\n\n"
                    f"Output:\n{result.output}"
                )

                # Verify the failure message IS shown (should be present on both buggy and fixed code)
                assert "Failed to cleanup" in result.output, (
                    f"Expected to see failure message in output.\n"
                    f"Output:\n{result.output}"
                )

    def test_cleanup_all_succeed_shows_only_success_message(self, runner, mock_sessions):
        """
        E2E Test: `pdd sessions cleanup --all --force` when all cleanups succeed.

        This test verifies the expected behavior when the operation actually succeeds.

        EXPECTED BEHAVIOR:
        - Output should show "✓ Successfully cleaned up 2 session(s)"
        - Output should NOT show any failure message
        - Exit code should be 0 to indicate success

        This test should PASS on both buggy and fixed code.
        """
        with patch("pdd.commands.sessions.CloudConfig") as mock_cloud_config:
            mock_cloud_config.get_jwt_token.return_value = "mock-jwt-token"

            with patch("pdd.commands.sessions.RemoteSessionManager") as mock_manager_class:
                # Mock list_sessions as a class/static method
                mock_manager_class.list_sessions = AsyncMock(return_value=mock_sessions)

                # Create mock instances for cleanup that always succeed
                async def successful_deregister():
                    return None  # Success (no exception)

                def create_successful_instance(*args, **kwargs):
                    instance = MagicMock()
                    instance.deregister = AsyncMock(side_effect=successful_deregister)
                    return instance

                mock_manager_class.side_effect = create_successful_instance

                result = runner.invoke(sessions, ["cleanup", "--all", "--force"])

                # Should exit with 0 when successful
                assert result.exit_code == 0, (
                    f"Exit code should be 0 when all cleanups succeed, got {result.exit_code}.\n"
                    f"Output:\n{result.output}"
                )

                # Should show success message
                assert "Successfully cleaned up" in result.output, (
                    f"Expected to see success message when cleanups succeed.\n"
                    f"Output:\n{result.output}"
                )

                # Should show count of 2 sessions
                assert "2 session(s)" in result.output or "2 session" in result.output, (
                    f"Expected to see count of 2 cleaned sessions.\n"
                    f"Output:\n{result.output}"
                )

                # Should NOT show "0 session(s)" in success message
                assert "Successfully cleaned up 0 session(s)" not in result.output, (
                    f"Success message should not show 0 sessions when cleanups succeeded.\n"
                    f"Output:\n{result.output}"
                )

    def test_cleanup_mixed_success_failure_exits_with_error(self, runner, mock_sessions):
        """
        E2E Test: `pdd sessions cleanup --all --force` with mixed success/failure.

        EXPECTED BEHAVIOR:
        - Output should show both success and failure messages
        - Exit code should be 1 (failure takes precedence)

        This verifies that ANY failure results in exit code 1.

        This test may FAIL on buggy code if exit code handling is missing.
        """
        with patch("pdd.commands.sessions.CloudConfig") as mock_cloud_config:
            mock_cloud_config.get_jwt_token.return_value = "mock-jwt-token"

            with patch("pdd.commands.sessions.RemoteSessionManager") as mock_manager_class:
                # Mock list_sessions as a class/static method
                mock_manager_class.list_sessions = AsyncMock(return_value=mock_sessions)

                # Create mock instances: first fails, second succeeds
                cleanup_call_count = [0]

                def create_instance(*args, **kwargs):
                    cleanup_call_count[0] += 1
                    instance = MagicMock()
                    if cleanup_call_count[0] == 1:
                        # First cleanup fails
                        async def failing_deregister():
                            raise Exception("Simulated network error")
                        instance.deregister = AsyncMock(side_effect=failing_deregister)
                    else:
                        # Second cleanup succeeds
                        async def successful_deregister():
                            return None
                        instance.deregister = AsyncMock(side_effect=successful_deregister)
                    return instance

                mock_manager_class.side_effect = create_instance

                result = runner.invoke(sessions, ["cleanup", "--all", "--force"])

                # Should exit with 1 when there are ANY failures
                assert result.exit_code == 1, (
                    f"BUG DETECTED (Issue #469): Exit code should be 1 when there are ANY failures, "
                    f"got {result.exit_code}.\n"
                    f"Mixed success/failure should be treated as a failure overall.\n"
                    f"Output:\n{result.output}"
                )

                # Should show both success and failure messages
                assert "Successfully cleaned up" in result.output, (
                    f"Expected to see success message for successful cleanups.\n"
                    f"Output:\n{result.output}"
                )

                assert "Failed to cleanup" in result.output, (
                    f"Expected to see failure message for failed cleanups.\n"
                    f"Output:\n{result.output}"
                )

    def test_cleanup_no_sessions_exits_successfully(self, runner):
        """
        E2E Test: `pdd sessions cleanup --all --force` when no sessions exist.

        EXPECTED BEHAVIOR:
        - Should show "No active remote sessions found"
        - Exit code should be 0 (this is not an error condition)

        This documents expected behavior for the edge case.
        """
        with patch("pdd.commands.sessions.CloudConfig") as mock_cloud_config:
            mock_cloud_config.get_jwt_token.return_value = "mock-jwt-token"

            with patch("pdd.commands.sessions.RemoteSessionManager") as mock_manager_class:
                # Mock list_sessions as a class/static method
                mock_manager_class.list_sessions = AsyncMock(return_value=[])

                result = runner.invoke(sessions, ["cleanup", "--all", "--force"])

                # Should exit with 0 (no sessions is not an error)
                assert result.exit_code == 0, (
                    f"Exit code should be 0 when no sessions exist, got {result.exit_code}.\n"
                    f"Output:\n{result.output}"
                )

                # Should show appropriate message
                assert "No active remote sessions found" in result.output, (
                    f"Expected to see message about no sessions.\n"
                    f"Output:\n{result.output}"
                )
