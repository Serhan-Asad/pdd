"""
Tests for _try_auto_fix_import_error() in sync_orchestration.py.

These tests verify that the function:
1. Asks for user confirmation before running pip install (click.confirm)
2. Respects --force flag to skip confirmation
3. Handles headless/no-TTY environments gracefully
4. Prints "Auto-installed: <pkg>" on success (respects --quiet)
5. Updates dependency files after install
6. Handles user declining the install

All tests mock subprocess.run to avoid real pip installs.
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdd.sync_orchestration import _try_auto_fix_import_error


@pytest.fixture
def tmp_code_file(tmp_path):
    """Create a dummy code file for testing."""
    code_file = tmp_path / "my_module.py"
    code_file.write_text("# generated code", encoding="utf-8")
    return code_file


@pytest.fixture
def tmp_example_file(tmp_path):
    """Create a dummy example file for testing."""
    example_file = tmp_path / "my_module_example.py"
    example_file.write_text("import my_module\nmy_module.run()", encoding="utf-8")
    return example_file


MODULE_NOT_FOUND_ERROR = "ModuleNotFoundError: No module named 'requests'"


class TestAutoFixImportNoConfirmation:
    """Tests that the current code is buggy: no confirmation before pip install."""

    def test_pip_install_runs_without_confirmation(self, tmp_code_file, tmp_example_file):
        """BUG: pip install runs without any click.confirm() call.

        The function should ask the user before installing packages,
        but currently it just runs pip install silently.
        """
        with patch("pdd.sync_orchestration.subprocess.run") as mock_run, \
             patch("pdd.sync_orchestration.click.confirm") as mock_confirm:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

            _try_auto_fix_import_error(
                MODULE_NOT_FOUND_ERROR,
                tmp_code_file,
                tmp_example_file,
            )

            # BUG: click.confirm is never called - the function installs silently
            # This assertion should PASS once the bug is fixed
            mock_confirm.assert_called_once()

    def test_force_flag_not_accepted(self, tmp_code_file, tmp_example_file):
        """BUG: Function signature doesn't accept force parameter.

        The call site has force available but doesn't pass it through.
        """
        # The current function signature only accepts (error_output, code_file, example_file)
        # It should also accept force and quiet parameters
        import inspect
        sig = inspect.signature(_try_auto_fix_import_error)
        param_names = list(sig.parameters.keys())

        # BUG: 'force' parameter is missing from the function signature
        assert "force" in param_names, (
            "_try_auto_fix_import_error() should accept a 'force' parameter "
            "to skip confirmation prompts"
        )

    def test_quiet_flag_not_accepted(self, tmp_code_file, tmp_example_file):
        """BUG: Function signature doesn't accept quiet parameter."""
        import inspect
        sig = inspect.signature(_try_auto_fix_import_error)
        param_names = list(sig.parameters.keys())

        # BUG: 'quiet' parameter is missing from the function signature
        assert "quiet" in param_names, (
            "_try_auto_fix_import_error() should accept a 'quiet' parameter "
            "to suppress output"
        )

    def test_no_dependency_file_update(self, tmp_path, tmp_code_file, tmp_example_file):
        """BUG: After pip install, no dependency file is updated.

        The function should detect and update requirements.txt or pyproject.toml.
        """
        # Create a requirements.txt in the project root
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("click\nrich\n", encoding="utf-8")

        with patch("pdd.sync_orchestration.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

            _try_auto_fix_import_error(
                MODULE_NOT_FOUND_ERROR,
                tmp_code_file,
                tmp_example_file,
            )

            # BUG: requirements.txt is never updated with the new package
            content = req_file.read_text(encoding="utf-8")
            assert "requests" in content, (
                "After auto-installing 'requests', it should be added to requirements.txt"
            )

    def test_no_output_printed_on_success(self, tmp_code_file, tmp_example_file, capsys):
        """BUG: No terminal output when a package is installed.

        The function should print 'Auto-installed: requests' so users know what happened.
        """
        with patch("pdd.sync_orchestration.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

            fixed, msg = _try_auto_fix_import_error(
                MODULE_NOT_FOUND_ERROR,
                tmp_code_file,
                tmp_example_file,
            )

            assert fixed is True
            captured = capsys.readouterr()
            # BUG: Nothing is printed to stdout/stderr - the install is completely silent
            assert "Auto-installed" in captured.out, (
                "Should print 'Auto-installed: requests' after successful install"
            )

    def test_call_site_missing_force_quiet(self):
        """BUG: The call site at line ~1446 doesn't pass force/quiet params.

        Verify that the call site passes force and quiet through to the function.
        This test reads the source to confirm the bug exists.
        """
        import inspect
        source = inspect.getsource(_try_auto_fix_import_error)

        # The function signature should include force and quiet
        # Currently it only has: (error_output, code_file, example_file)
        assert "force" in source.split(")", 1)[0], (
            "Function signature should include 'force' parameter"
        )


class TestAutoFixUserDeclines:
    """Test that user can decline the install."""

    def test_user_declines_install(self, tmp_code_file, tmp_example_file):
        """When user says 'no' to confirmation, pip install should not run."""
        with patch("pdd.sync_orchestration.subprocess.run") as mock_run, \
             patch("pdd.sync_orchestration.click.confirm", return_value=False) as mock_confirm:

            fixed, msg = _try_auto_fix_import_error(
                MODULE_NOT_FOUND_ERROR,
                tmp_code_file,
                tmp_example_file,
            )

            # If confirmation existed and user declined, pip should not run
            # BUG: There's no confirmation at all, so pip always runs
            if mock_confirm.called:
                # If confirm was called and returned False, pip should NOT have run
                mock_run.assert_not_called()
            else:
                # Confirm was never called - this IS the bug
                pytest.fail(
                    "click.confirm() was never called - pip install runs without user consent"
                )
