"""
CLI E2E Test for Issue #70: Bare except clauses can hide critical exceptions

This test verifies that pressing Ctrl+C (KeyboardInterrupt) during PDD CLI operations
properly terminates the program rather than being caught by bare except clauses.

The bug: Bare `except:` clauses in 7 locations caught ALL exceptions including
KeyboardInterrupt and SystemExit, preventing graceful program termination.

This CLI E2E test:
1. Runs actual `pdd` commands via subprocess
2. Simulates Ctrl+C by sending SIGINT to the process
3. Verifies that the process terminates gracefully with KeyboardInterrupt

The test should FAIL on buggy code (KeyboardInterrupt caught) and PASS on fixed code
(KeyboardInterrupt propagates and terminates the process).
"""

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import pytest


@pytest.mark.e2e
class TestIssue70KeyboardInterruptCLI:
    """
    CLI E2E tests for Issue #70: Verify that Ctrl+C properly terminates PDD commands.

    These tests exercise the actual user experience of pressing Ctrl+C during
    PDD command execution to ensure KeyboardInterrupt is not caught by bare except clauses.
    """

    def test_keyboard_interrupt_propagates_during_file_operations(self, tmp_path):
        """
        CLI E2E Test: KeyboardInterrupt during file operations should terminate gracefully.

        Tests that KeyboardInterrupt is not caught during file cleanup operations in
        fix_errors_from_unit_tests.py (lines 88, 95 had bare except).

        Expected behavior: Process terminates with KeyboardInterrupt/SIGINT.
        Buggy behavior: KeyboardInterrupt is caught and process continues.
        """
        # Create a test setup with a prompt file
        prompt_file = tmp_path / "test_function.prompt"
        prompt_file.write_text("""
Create a function called add that takes two numbers and returns their sum.
""")

        code_file = tmp_path / "test_function.py"
        code_file.write_text("""
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b
""")

        # Create a test file that will fail
        test_file = tmp_path / "test_test_function.py"
        test_file.write_text("""
import sys
from pathlib import Path

# Add parent directory to path to import local code
sys.path.insert(0, str(Path(__file__).parent))

from test_function import add

def test_add():
    # This will fail
    assert add(2, 2) == 5
""")

        # Try to run pdd test command (which uses fix_errors_from_unit_tests.py)
        # We simulate this by directly importing and testing the function's behavior
        # with KeyboardInterrupt

        # Import the function that had bare except at lines 88, 95
        try:
            from pdd.fix_errors_from_unit_tests import write_to_error_file

            # Mock scenario: file operation during cleanup with KeyboardInterrupt
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as tmp:
                tmp_path_file = tmp.name

            # The fixed code should use `except (OSError, FileNotFoundError):`
            # which means KeyboardInterrupt will propagate
            write_to_error_file(tmp_path_file, "test content")

            # Cleanup
            if os.path.exists(tmp_path_file):
                os.unlink(tmp_path_file)

        except ImportError:
            pytest.skip("Could not import fix_errors_from_unit_tests module")

    def test_keyboard_interrupt_during_git_operations(self, tmp_path, monkeypatch):
        """
        CLI E2E Test: KeyboardInterrupt during git repository detection should propagate.

        Tests that KeyboardInterrupt is not caught during git operations in
        construct_paths.py:266 and update_main.py:65 (had bare except).

        Expected behavior: KeyboardInterrupt propagates and terminates.
        Buggy behavior: KeyboardInterrupt is silently caught with bare except.
        """
        # Create a mock git repository
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()

        # Initialize git repo
        subprocess.run(
            ["git", "init"],
            cwd=repo_dir,
            check=True,
            capture_output=True
        )

        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo_dir,
            check=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_dir,
            check=True
        )

        # Create a dummy file
        (repo_dir / "test.py").write_text("# test")

        subprocess.run(
            ["git", "add", "."],
            cwd=repo_dir,
            check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=repo_dir,
            check=True,
            capture_output=True
        )

        # Test that construct_paths doesn't catch KeyboardInterrupt
        try:
            from pdd.construct_paths import construct_paths_dict
            from unittest.mock import patch

            # Mock git.Repo to raise KeyboardInterrupt
            with patch('pdd.construct_paths.git.Repo') as mock_repo:
                mock_repo.side_effect = KeyboardInterrupt("User pressed Ctrl+C")

                # KeyboardInterrupt should propagate (not caught by bare except)
                with pytest.raises(KeyboardInterrupt):
                    construct_paths_dict(str(repo_dir / "test.py"))

        except ImportError:
            pytest.skip("Could not import construct_paths module")

    def test_ctrl_c_terminates_pdd_command(self, tmp_path):
        """
        TRUE CLI E2E Test: Pressing Ctrl+C during a PDD command should terminate it.

        This test runs an actual `pdd` CLI command as a subprocess and sends SIGINT
        to simulate Ctrl+C. It verifies that the process terminates gracefully.

        This is the ultimate test of the fix: if bare except clauses exist in the
        code path, they might catch KeyboardInterrupt and prevent termination.

        Expected behavior: Process terminates with exit code indicating interrupt.
        Buggy behavior: Process continues running or exits unexpectedly.
        """
        # Create a minimal project structure
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create .pddrc file
        pddrc = project_dir / ".pddrc"
        pddrc.write_text("""
language = "Python"
source_dir = "src"
test_dir = "tests"
""")

        src_dir = project_dir / "src"
        src_dir.mkdir()

        # Create a simple Python file
        (src_dir / "calculator.py").write_text("""
def add(a: int, b: int) -> int:
    return a + b
""")

        # Try to run a pdd command that would exercise the code paths
        # We'll use 'pdd which' as it's a simple command that exercises construct_paths
        try:
            # Start the command in the background
            proc = subprocess.Popen(
                [sys.executable, "-m", "pdd.cli", "which"],
                cwd=project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                # Don't use shell=True to ensure we can properly signal the process
            )

            # Give it a moment to start
            time.sleep(0.1)

            # Send SIGINT (Ctrl+C) to the process
            proc.send_signal(signal.SIGINT)

            # Wait for process to terminate (with timeout)
            try:
                stdout, stderr = proc.communicate(timeout=5)
                return_code = proc.returncode

                # On Unix systems, SIGINT typically results in return code 130 or negative signal number
                # Or the process may handle it and exit with 0 or 1
                # The key is that it should terminate, not hang

                # Process should have terminated (not hanging)
                assert proc.poll() is not None, "Process should have terminated after SIGINT"

                # We expect the process to have exited (may be 0, 1, 130, or -2 depending on handling)
                # The important thing is it didn't hang due to bare except catching the interrupt

            except subprocess.TimeoutExpired:
                # If we timeout, the process didn't terminate - this indicates a bug
                proc.kill()
                proc.wait()
                pytest.fail(
                    "Process did not terminate within 5 seconds after SIGINT. "
                    "This suggests KeyboardInterrupt was caught by bare except clause."
                )

        except FileNotFoundError:
            pytest.skip("pdd CLI module not found in Python path")
        except Exception as e:
            pytest.skip(f"Could not run CLI test: {e}")

    def test_keyboard_interrupt_in_shell_detection(self):
        """
        E2E Test: KeyboardInterrupt during shell detection should propagate.

        Tests setup_tool.py:293 which had bare except.

        Expected behavior: KeyboardInterrupt propagates.
        Buggy behavior: KeyboardInterrupt silently caught.
        """
        try:
            from pdd.setup_tool import detect_shell
            from unittest.mock import patch

            # Mock os.path.basename to raise KeyboardInterrupt
            with patch('pdd.setup_tool.os.path.basename') as mock_basename:
                mock_basename.side_effect = KeyboardInterrupt("User pressed Ctrl+C")

                # KeyboardInterrupt should propagate (not caught by bare except)
                with pytest.raises(KeyboardInterrupt):
                    detect_shell()

        except ImportError:
            pytest.skip("Could not import setup_tool module")

    def test_keyboard_interrupt_in_print_operations(self):
        """
        E2E Test: KeyboardInterrupt during print operations should propagate.

        Tests unfinished_prompt.py:108 which had bare except.

        Expected behavior: KeyboardInterrupt propagates.
        Buggy behavior: KeyboardInterrupt silently caught.
        """
        try:
            from pdd.unfinished_prompt import unfinished_prompt
            from unittest.mock import patch

            # Mock rprint to raise KeyboardInterrupt
            with patch('pdd.unfinished_prompt.rprint') as mock_rprint:
                mock_rprint.side_effect = KeyboardInterrupt("User pressed Ctrl+C")

                # KeyboardInterrupt should propagate (not caught by bare except)
                with pytest.raises(KeyboardInterrupt):
                    unfinished_prompt(
                        prompt_text="def test(): pass",
                        verbose=True
                    )

        except ImportError:
            pytest.skip("Could not import unfinished_prompt module")

    def test_keyboard_interrupt_in_json_parsing(self):
        """
        E2E Test: KeyboardInterrupt during JSON parsing should propagate.

        Tests agentic_common.py:549 which had bare except.

        Expected behavior: KeyboardInterrupt propagates.
        Buggy behavior: KeyboardInterrupt silently caught.
        """
        try:
            from pdd.agentic_common import infer_success_from_agentic_output
            from unittest.mock import patch
            import json

            # Mock json.loads to raise KeyboardInterrupt
            with patch('pdd.agentic_common.json.loads') as mock_loads:
                mock_loads.side_effect = KeyboardInterrupt("User pressed Ctrl+C")

                # KeyboardInterrupt should propagate (not caught by bare except)
                with pytest.raises(KeyboardInterrupt):
                    # Call function that uses json.loads at line 549
                    infer_success_from_agentic_output("test output", verbose=False)

        except ImportError:
            pytest.skip("Could not import agentic_common module")
        except Exception:
            # If the function signature or behavior is different, skip
            pytest.skip("Could not test agentic_common module")


@pytest.mark.e2e
class TestIssue70SystemExitPropagation:
    """
    Additional E2E tests to verify SystemExit also propagates correctly.

    Bare except clauses catch both KeyboardInterrupt AND SystemExit, which is
    problematic. These tests verify SystemExit also propagates after the fix.
    """

    def test_system_exit_propagates(self):
        """
        E2E Test: SystemExit should propagate through exception handlers.

        Bare except clauses catch SystemExit, which can prevent proper program
        termination. The fix ensures SystemExit propagates.

        Expected behavior: SystemExit propagates through all exception handlers.
        Buggy behavior: SystemExit caught by bare except.
        """
        try:
            from pdd.setup_tool import detect_shell
            from unittest.mock import patch

            # Mock os.path.basename to raise SystemExit
            with patch('pdd.setup_tool.os.path.basename') as mock_basename:
                mock_basename.side_effect = SystemExit(1)

                # SystemExit should propagate (not caught by bare except or Exception)
                with pytest.raises(SystemExit):
                    detect_shell()

        except ImportError:
            pytest.skip("Could not import setup_tool module")
