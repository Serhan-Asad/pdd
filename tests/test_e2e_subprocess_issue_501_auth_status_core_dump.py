"""
E2E Test (subprocess-based) for Issue #501: pdd auth status treats
'not authenticated' as unexpected error and writes unnecessary core dump.

This is a true E2E test that uses subprocess to invoke the actual CLI binary,
exercising the full code path that a user would take. No mocking of the buggy
component (PDDCLI.invoke exception handling).

Bug: Running `pdd auth status` when not authenticated displays:
  Not authenticated.
  Run: pdd auth login
  Error during 'unknown' command:
    An unexpected error occurred: Process exited with code 1
  Debug snapshot saved to .pdd/core_dumps/pdd-core-<timestamp>.json

Root Cause: sys.exit(1) in auth.py:200 raises SystemExit, which
PDDCLI.invoke() at cli.py:116-122 catches and converts to
RuntimeError("Process exited with code 1"). This flows to handle_error()
which prints "An unexpected error occurred" and triggers a core dump write.

Expected: Clean "Not authenticated." message, exit code 1, no error text,
no core dump file.

E2E Test Strategy:
- Use subprocess to run the CLI in isolation (like a real user)
- Set HOME to a temp directory with no JWT cache (simulating not authenticated)
- Verify the output does NOT contain "unexpected error" or "Error during"
- Verify no core dump files are created in .pdd/core_dumps/
- This exercises the real code path without mocking the buggy component

Issue: https://github.com/gltanaka/pdd/issues/501
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

import pytest


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "pdd").is_dir() and (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root with pdd/ directory")


def _run_pdd_command(
    command_args: list,
    home_dir: Path,
    timeout: int = 30
) -> Tuple[int, str, str]:
    """
    Run a pdd command with custom HOME directory via subprocess.

    Returns (return_code, stdout, stderr).
    """
    project_root = get_project_root()

    env = os.environ.copy()
    # Set HOME so JWT_CACHE_FILE (~/.pdd/jwt_cache) points to our test directory
    env["HOME"] = str(home_dir)
    env["PYTHONPATH"] = str(project_root)
    # Disable auto-update to avoid network calls
    env["PDD_AUTO_UPDATE"] = "false"
    # Suppress onboarding reminder
    env["PDD_SUPPRESS_SETUP_REMINDER"] = "1"

    result = subprocess.run(
        [sys.executable, "-m", "pdd.cli"] + command_args,
        capture_output=True,
        text=True,
        cwd=str(project_root),
        env=env,
        timeout=timeout
    )

    return result.returncode, result.stdout, result.stderr


class TestIssue501E2ESubprocess:
    """
    E2E tests using subprocess to verify the auth status core dump bug.

    These tests exercise the full CLI path that users take when running
    `pdd auth status` when not authenticated. No mocking — the real
    PDDCLI.invoke() exception handling is exercised.
    """

    def test_auth_status_unauthenticated_no_unexpected_error_message(self, tmp_path: Path):
        """
        E2E Test: `pdd auth status` when not authenticated must not print
        'An unexpected error occurred'.

        User scenario:
        1. User has never authenticated (no JWT cache)
        2. User runs `pdd auth status`

        Expected behavior:
        - Output shows "Not authenticated."
        - Output shows "Run: pdd auth login"
        - Exit code is 1
        - NO "An unexpected error occurred" message
        - NO "Error during" message

        Actual buggy behavior:
        - All of the above PLUS:
        - "Error during 'unknown' command:"
        - "  An unexpected error occurred: Process exited with code 1"
        - "Debug snapshot saved to .pdd/core_dumps/..."

        This test FAILS on buggy code, PASSES after fix.
        """
        # Setup: Create home directory with no JWT cache (not authenticated)
        home_dir = tmp_path / "home"
        pdd_dir = home_dir / ".pdd"
        pdd_dir.mkdir(parents=True, exist_ok=True)

        # Run the auth status command
        returncode, stdout, stderr = _run_pdd_command(
            ["auth", "status"],
            home_dir=home_dir
        )

        full_output = stdout + stderr

        # Verify the informational message IS present
        assert "Not authenticated" in full_output, (
            f"Expected 'Not authenticated' in output.\n"
            f"stdout: {stdout}\nstderr: {stderr}"
        )

        # Verify exit code is 1 (expected for not authenticated)
        assert returncode == 1, (
            f"Expected exit code 1, got {returncode}\n"
            f"stdout: {stdout}\nstderr: {stderr}"
        )

        # THE BUG: These messages should NOT appear
        # This assertion FAILS on buggy code, PASSES after fix
        assert "unexpected error" not in full_output.lower(), (
            f"BUG DETECTED (Issue #501): 'An unexpected error occurred' message "
            f"was printed for a normal 'not authenticated' status check.\n\n"
            f"This is misleading — exit code 1 is expected behavior for "
            f"'pdd auth status' when not authenticated.\n\n"
            f"Root cause: PDDCLI.invoke() at cli.py:116-122 converts "
            f"SystemExit(1) to RuntimeError, which triggers handle_error().\n\n"
            f"Return code: {returncode}\n"
            f"Full output:\n{full_output}"
        )

        assert "Error during" not in full_output, (
            f"BUG DETECTED (Issue #501): 'Error during' message was printed "
            f"for a normal 'not authenticated' status check.\n\n"
            f"Full output:\n{full_output}"
        )

    def test_auth_status_unauthenticated_no_core_dump_written(self, tmp_path: Path):
        """
        E2E Test: `pdd auth status` when not authenticated must not write
        a core dump file to disk.

        User scenario:
        1. User has never authenticated
        2. User runs `pdd auth status`
        3. No core dump should be written or mentioned in output

        Actual buggy behavior:
        - "Debug snapshot saved to .pdd/core_dumps/pdd-core-<timestamp>.json"
          is printed to the terminal
        - A core dump JSON file is written to .pdd/core_dumps/

        This test FAILS on buggy code, PASSES after fix.
        """
        # Setup: Create home directory with no JWT cache
        home_dir = tmp_path / "home"
        pdd_dir = home_dir / ".pdd"
        pdd_dir.mkdir(parents=True, exist_ok=True)

        # Run the auth status command
        returncode, stdout, stderr = _run_pdd_command(
            ["auth", "status"],
            home_dir=home_dir
        )

        full_output = stdout + stderr

        # THE BUG: "Debug snapshot saved" message should NOT appear for expected exits
        # Core dumps are written to CWD/.pdd/core_dumps/, but the message in
        # stdout/stderr is the reliable indicator of whether a dump was triggered.
        assert "Debug snapshot saved" not in full_output, (
            f"BUG DETECTED (Issue #501): A core dump was written for a normal "
            f"'not authenticated' status check.\n\n"
            f"Core dumps should only be created for actual unexpected errors, "
            f"not for expected exit code 1 from 'pdd auth status'.\n\n"
            f"Full output:\n{full_output}"
        )

    def test_auth_status_unauthenticated_no_unknown_command_name(self, tmp_path: Path):
        """
        E2E Test: Error output (if any) should not show 'unknown' as command name.

        The secondary bug is that if the error handler IS triggered, it shows
        "Error during 'unknown' command" instead of "Error during 'auth status'".

        This test verifies that 'unknown' does not appear in the output.

        This test FAILS on buggy code, PASSES after fix.
        """
        # Setup: Create home directory with no JWT cache
        home_dir = tmp_path / "home"
        pdd_dir = home_dir / ".pdd"
        pdd_dir.mkdir(parents=True, exist_ok=True)

        # Run the auth status command
        returncode, stdout, stderr = _run_pdd_command(
            ["auth", "status"],
            home_dir=home_dir
        )

        full_output = stdout + stderr

        # THE BUG: "Error during 'unknown' command" should never appear
        assert "'unknown'" not in full_output, (
            f"BUG DETECTED (Issue #501): Command name shows as 'unknown' instead "
            f"of 'auth status' in error output.\n\n"
            f"Root cause: _first_pending_command() in utils.py reads "
            f"ctx.protected_args which is empty for nested subcommands.\n\n"
            f"Full output:\n{full_output}"
        )

    def test_auth_status_unauthenticated_clean_output_only(self, tmp_path: Path):
        """
        E2E Test: `pdd auth status` output should contain ONLY the expected
        informational messages, nothing else.

        This is a strict test that verifies the output is exactly what the
        user should see: "Not authenticated." and "Run: pdd auth login".

        This test FAILS on buggy code (extra error lines), PASSES after fix.
        """
        # Setup: Create home directory with no JWT cache
        home_dir = tmp_path / "home"
        pdd_dir = home_dir / ".pdd"
        pdd_dir.mkdir(parents=True, exist_ok=True)

        # Run the auth status command
        returncode, stdout, stderr = _run_pdd_command(
            ["auth", "status"],
            home_dir=home_dir
        )

        full_output = stdout + stderr

        # Expected lines (from auth.py:198-199)
        assert "Not authenticated" in full_output
        assert "pdd auth login" in full_output

        # Strip ANSI codes and whitespace for line-by-line analysis
        import re
        clean_output = re.sub(r'\x1b\[[0-9;]*m', '', full_output).strip()
        lines = [line.strip() for line in clean_output.splitlines() if line.strip()]

        # THE BUG: On buggy code, there are extra lines from the error handler.
        # We expect ONLY the two informational lines, nothing more.
        # Filter out any non-bug-related lines (logging, update notices, etc.)
        unexpected_lines = []
        for line in lines:
            # Skip expected informational lines
            if "Not authenticated" in line:
                continue
            if "pdd auth login" in line:
                continue
            # Skip any version/update notices (not part of the bug)
            if "update" in line.lower() and "error" not in line.lower():
                continue
            if "version" in line.lower() and "error" not in line.lower():
                continue
            # Skip setup reminder lines
            if "setup" in line.lower() and "error" not in line.lower():
                continue
            # Skip Python logging lines (INFO, DEBUG, WARNING from loggers)
            if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - ', line):
                continue
            # Anything else is unexpected
            unexpected_lines.append(line)

        assert len(unexpected_lines) == 0, (
            f"BUG DETECTED (Issue #501): Unexpected lines in 'pdd auth status' output.\n\n"
            f"Expected only: 'Not authenticated.' and 'Run: pdd auth login'\n"
            f"Unexpected lines:\n"
            + "\n".join(f"  > {line}" for line in unexpected_lines)
            + f"\n\nFull output:\n{full_output}"
        )


class TestIssue501E2EWithCoreDumpFlag:
    """
    E2E tests verifying the --no-core-dump flag interaction with the bug.

    Even with --no-core-dump, the error MESSAGE still appears.
    """

    def test_auth_status_with_no_core_dump_flag_still_shows_error(self, tmp_path: Path):
        """
        E2E Test: Even with --no-core-dump, the misleading error message appears.

        The --no-core-dump flag suppresses the core dump FILE, but the error
        handler still prints "An unexpected error occurred" to the terminal.

        This test FAILS on buggy code, PASSES after fix.
        """
        # Setup: Create home directory with no JWT cache
        home_dir = tmp_path / "home"
        pdd_dir = home_dir / ".pdd"
        pdd_dir.mkdir(parents=True, exist_ok=True)

        # Run with --no-core-dump flag
        returncode, stdout, stderr = _run_pdd_command(
            ["--no-core-dump", "auth", "status"],
            home_dir=home_dir
        )

        full_output = stdout + stderr

        # Verify expected content
        assert "Not authenticated" in full_output

        # THE BUG: Error message appears even with --no-core-dump
        # The flag only suppresses the dump file, not the error message
        assert "unexpected error" not in full_output.lower(), (
            f"BUG DETECTED (Issue #501): Even with --no-core-dump, "
            f"the 'unexpected error' message is printed.\n\n"
            f"Full output:\n{full_output}"
        )
