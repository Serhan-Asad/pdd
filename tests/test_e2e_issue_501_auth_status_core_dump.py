"""
Issue #501: pdd auth status treats 'not authenticated' as unexpected error
and writes unnecessary core dump.

These tests invoke through the full CLI (not auth_group directly)
because the bug lives in PDDCLI.invoke() exception handling.

Bug: sys.exit(1) in auth.py raises SystemExit, which PDDCLI.invoke()
catches and converts to RuntimeError -> handle_error() prints
"An unexpected error occurred" + writes a core dump.

Expected: Clean "Not authenticated." message, exit code 1, no error text.
"""
import json
import os
import time
from unittest.mock import MagicMock, patch, AsyncMock, call

import click
import pytest
from click.testing import CliRunner

# Import the fully-registered CLI (has auth command) for integration tests
from pdd.cli import cli as cli_command
# Import PDDCLI class for unit tests (tests 6 & 7)
from pdd.core.cli import PDDCLI


@pytest.fixture
def runner():
    return CliRunner(mix_stderr=False)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure clean environment for all tests."""
    # Disable auto-update to avoid network calls
    monkeypatch.setenv("PDD_AUTO_UPDATE", "false")
    # Suppress onboarding reminder
    monkeypatch.setenv("PDD_SUPPRESS_SETUP_REMINDER", "1")


# ---------------------------------------------------------------------------
# Test 1: Primary bug -- "unexpected error" message should not appear
# ---------------------------------------------------------------------------
class TestAuthStatusUnauthenticatedNoUnexpectedError:
    """When not authenticated, 'pdd auth status' should NOT print
    'An unexpected error occurred'."""

    def test_auth_status_unauthenticated_no_unexpected_error(self, runner, monkeypatch):
        """Primary bug test: 'unexpected error' message must not appear for
        a normal 'not authenticated' status check."""
        with patch("pdd.commands.auth.get_auth_status") as mock_status, \
             patch("pdd.core.cli.auto_update"), \
             patch("pdd.core.dump.garbage_collect_core_dumps"):
            mock_status.return_value = {"authenticated": False}

            result = runner.invoke(cli_command, ["--no-core-dump", "auth", "status"])

        # Should show the informational message
        assert "Not authenticated" in result.output

        # BUG: These assertions fail on buggy code because PDDCLI.invoke()
        # converts SystemExit(1) to RuntimeError and calls handle_error()
        assert "unexpected error" not in result.output.lower(), \
            "Bug #501: 'unexpected error' should not appear for normal auth status check"
        assert "Error during" not in result.output, \
            "Bug #501: error handler should not be triggered for expected exit code 1"


# ---------------------------------------------------------------------------
# Test 2: No core dump file should be written for expected exits
# ---------------------------------------------------------------------------
class TestAuthStatusUnauthenticatedNoCoreDumpWrite:
    """Core dump should NOT be written when 'pdd auth status' exits with
    code 1 for a normal 'not authenticated' condition."""

    def test_auth_status_unauthenticated_no_core_dump_write(self, runner, monkeypatch):
        """No core dump should be written for expected 'not authenticated' exit."""
        with patch("pdd.commands.auth.get_auth_status") as mock_status, \
             patch("pdd.core.cli.auto_update"), \
             patch("pdd.core.dump.garbage_collect_core_dumps"), \
             patch("pdd.core.cli._write_core_dump") as mock_dump:
            mock_status.return_value = {"authenticated": False}

            result = runner.invoke(cli_command, ["--no-core-dump", "auth", "status"])

        # BUG: On buggy code, _write_core_dump IS called from the exception
        # handler path in PDDCLI.invoke() because SystemExit(1) is caught
        # and treated as an error.
        # When the bug is fixed, SystemExit(1) will be re-raised and the
        # exception handler path (which calls _write_core_dump) won't execute.
        mock_dump.assert_not_called(), \
            "Bug #501: core dump should not be written for expected 'not authenticated' exit"


# ---------------------------------------------------------------------------
# Test 3: handle_error() should NOT be called for expected exits
# ---------------------------------------------------------------------------
class TestAuthStatusUnauthenticatedHandleErrorNotCalled:
    """handle_error() should never be reached for expected exit code 1."""

    def test_auth_status_unauthenticated_handle_error_not_called(self, runner, monkeypatch):
        """handle_error() must not be called for expected 'not authenticated' exit."""
        with patch("pdd.commands.auth.get_auth_status") as mock_status, \
             patch("pdd.core.cli.auto_update"), \
             patch("pdd.core.dump.garbage_collect_core_dumps"), \
             patch("pdd.core.cli.handle_error") as mock_handle_error:
            mock_status.return_value = {"authenticated": False}

            result = runner.invoke(cli_command, ["--no-core-dump", "auth", "status"])

        # BUG: On buggy code, handle_error IS called because SystemExit(1)
        # is converted to RuntimeError and routed through the error handler.
        mock_handle_error.assert_not_called(), \
            "Bug #501: handle_error() should not be called for expected exit code 1"


# ---------------------------------------------------------------------------
# Test 4: Positive control -- authenticated status exits cleanly
# ---------------------------------------------------------------------------
class TestAuthStatusAuthenticatedExitsCleanly:
    """When authenticated, 'pdd auth status' should exit cleanly (code 0)
    with no error messages."""

    def test_auth_status_authenticated_exits_cleanly(self, runner, monkeypatch):
        """Positive control: authenticated status should work cleanly through PDDCLI."""
        with patch("pdd.commands.auth.get_auth_status") as mock_status, \
             patch("pdd.commands.auth.JWT_CACHE_FILE") as mock_cache, \
             patch("pdd.core.cli.auto_update"), \
             patch("pdd.core.dump.garbage_collect_core_dumps"), \
             patch("pdd.core.cli.handle_error") as mock_handle_error:
            mock_status.return_value = {"authenticated": True, "cached": True}
            mock_cache.exists.return_value = True
            mock_cache.read_text.return_value = json.dumps({
                "id_token": "header.eyJlbWFpbCI6InRlc3RAZXhhbXBsZS5jb20ifQ.sig",
            })

            result = runner.invoke(cli_command, ["--no-core-dump", "auth", "status"])

        # Authenticated path uses sys.exit(0) which raises SystemExit(0),
        # which PDDCLI.invoke() correctly re-raises. So this should work.
        assert "unexpected error" not in result.output.lower()
        mock_handle_error.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5: --verify failure path also affected
# ---------------------------------------------------------------------------
class TestAuthStatusVerifyFailureNoUnexpectedError:
    """When --verify detects auth failure, the error message should be clean
    (no 'unexpected error' wrapper)."""

    def test_auth_status_verify_failure_no_unexpected_error(self, runner, monkeypatch):
        """--verify failure should not trigger 'unexpected error' through PDDCLI."""
        with patch("pdd.commands.auth.get_auth_status") as mock_status, \
             patch("pdd.commands.auth.JWT_CACHE_FILE") as mock_cache, \
             patch("pdd.commands.auth.verify_auth", new_callable=AsyncMock) as mock_verify, \
             patch("pdd.core.cli.auto_update"), \
             patch("pdd.core.dump.garbage_collect_core_dumps"):
            mock_status.return_value = {
                "authenticated": True,
                "cached": False,
                "expires_at": None,
            }
            mock_cache.exists.return_value = True
            mock_cache.read_text.return_value = json.dumps({
                "id_token": "header.eyJlbWFpbCI6InRlc3RAZXhhbXBsZS5jb20ifQ.sig",
            })
            mock_verify.return_value = {
                "valid": False,
                "error": "Invalid refresh token",
                "needs_reauth": True,
                "username": None,
            }

            result = runner.invoke(cli_command, ["--no-core-dump", "auth", "status", "--verify"])

        # Should show verification failure message
        assert "Authentication verification failed" in result.output

        # BUG: sys.exit(1) at auth.py:264 also triggers the same PDDCLI bug
        assert "unexpected error" not in result.output.lower(), \
            "Bug #501: --verify failure should not trigger 'unexpected error'"
        assert "Error during" not in result.output, \
            "Bug #501: error handler should not wrap --verify failure"


# ---------------------------------------------------------------------------
# Test 6: PDDCLI re-raises SystemExit instead of converting to RuntimeError
# ---------------------------------------------------------------------------
class TestPDDCLIReRaisesSystemExit:
    """Unit test: PDDCLI.invoke() should re-raise SystemExit (all codes)
    instead of converting non-zero SystemExit to RuntimeError."""

    def test_pddcli_reraises_system_exit_instead_of_converting(self):
        """PDDCLI.invoke() should not convert SystemExit(1) to RuntimeError.

        Currently (buggy): SystemExit(1) -> RuntimeError('Process exited with code 1')
        Expected (fixed): SystemExit(1) is re-raised directly.
        """
        # Create a minimal Click command that raises SystemExit(1)
        @click.command("test-exit")
        def exit_cmd():
            raise SystemExit(1)

        # Create a PDDCLI group and add the command
        group = PDDCLI(name="test-cli")
        group.add_command(exit_cmd)

        runner = CliRunner(mix_stderr=False)
        with patch("pdd.core.cli.handle_error") as mock_handle_error, \
             patch("pdd.core.cli._write_core_dump"):
            result = runner.invoke(group, ["test-exit"])

        # BUG: On buggy code, handle_error IS called because SystemExit(1)
        # is converted to RuntimeError. When fixed, SystemExit(1) should
        # propagate without triggering handle_error.
        mock_handle_error.assert_not_called(), \
            "Bug #501: PDDCLI should re-raise SystemExit(1), not convert to RuntimeError"

        # The exit code should still be 1
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Test 7: PDDCLI re-raises click.exceptions.Exit instead of converting
# ---------------------------------------------------------------------------
class TestPDDCLIReRaisesClickExit:
    """Unit test: PDDCLI.invoke() should re-raise click.exceptions.Exit
    (all codes) instead of converting non-zero to RuntimeError."""

    def test_pddcli_reraises_click_exit_instead_of_converting(self):
        """PDDCLI.invoke() should not convert click.exceptions.Exit(1) to RuntimeError.

        Currently (buggy): Exit(1) -> RuntimeError('Command exited with code 1')
        Expected (fixed): Exit(1) is re-raised directly.
        """
        # Create a minimal Click command that raises click.exceptions.Exit(1)
        @click.command("test-click-exit")
        def click_exit_cmd():
            raise click.exceptions.Exit(1)

        # Create a PDDCLI group and add the command
        group = PDDCLI(name="test-cli")
        group.add_command(click_exit_cmd)

        runner = CliRunner(mix_stderr=False)
        with patch("pdd.core.cli.handle_error") as mock_handle_error, \
             patch("pdd.core.cli._write_core_dump"):
            result = runner.invoke(group, ["test-click-exit"])

        # BUG: On buggy code, handle_error IS called because Exit(1)
        # is converted to RuntimeError. When fixed, Exit(1) should
        # propagate without triggering handle_error.
        mock_handle_error.assert_not_called(), \
            "Bug #501: PDDCLI should re-raise Exit(1), not convert to RuntimeError"

        # The exit code should still be 1
        assert result.exit_code == 1
