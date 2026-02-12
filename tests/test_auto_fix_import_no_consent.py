"""Tests for issue #494: _try_auto_fix_import_error must require user consent before pip install.

The bug: _try_auto_fix_import_error() in both sync_orchestration.py and pin_example_hack.py
calls subprocess.run([..., 'pip', 'install', package]) unconditionally — no click.confirm(),
no --force check, no allowlist.

These tests assert the CORRECT behavior: pip install should only run after user confirmation.
They FAIL on the current buggy code and PASS once the fix is applied.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture
def import_error_setup(tmp_path):
    """Create minimal files needed to trigger the external package pip install path."""
    code_file = tmp_path / "calculator.py"
    code_file.write_text("# code file\n")
    example_file = tmp_path / "example_calculator.py"
    example_file.write_text("import numpy\nprint('hello')\n")
    error_output = "ModuleNotFoundError: No module named 'numpy'"
    return code_file, example_file, error_output


class TestSyncOrchestrationAutoFixConsent:
    """_try_auto_fix_import_error in sync_orchestration.py should ask before pip install."""

    def test_pip_install_requires_user_confirmation(self, import_error_setup):
        """pip install should NOT run without user confirmation via click.confirm().

        Expected (fixed) behavior: click.confirm() is called before pip install.
        Current (buggy) behavior: pip install runs unconditionally, click.confirm() is never called.
        """
        code_file, example_file, error_output = import_error_setup

        from pdd.sync_orchestration import _try_auto_fix_import_error

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch('pdd.sync_orchestration.subprocess.run', return_value=mock_result) as mock_run, \
             patch('pdd.sync_orchestration.click') as mock_click:
            mock_click.confirm.return_value = True
            fixed, msg = _try_auto_fix_import_error(error_output, code_file, example_file)

        # ASSERT: click.confirm must be called before pip install
        # FAILS on buggy code because click.confirm is never called
        mock_click.confirm.assert_called_once()

    def test_pip_install_skipped_when_user_declines(self, import_error_setup):
        """When user declines confirmation, pip install should NOT run."""
        code_file, example_file, error_output = import_error_setup

        from pdd.sync_orchestration import _try_auto_fix_import_error

        with patch('pdd.sync_orchestration.subprocess.run') as mock_run, \
             patch('pdd.sync_orchestration.click') as mock_click:
            mock_click.confirm.return_value = False
            fixed, msg = _try_auto_fix_import_error(error_output, code_file, example_file)

        # ASSERT: pip install should not run when user declines
        # FAILS on buggy code because there's no confirmation gate — pip always runs
        mock_run.assert_not_called()

    def test_function_accepts_force_parameter(self):
        """The function should accept a force parameter to bypass confirmation in CI."""
        from pdd.sync_orchestration import _try_auto_fix_import_error
        import inspect

        sig = inspect.signature(_try_auto_fix_import_error)
        # FAILS on buggy code: no force parameter exists
        assert 'force' in sig.parameters, \
            "_try_auto_fix_import_error should accept a 'force' parameter"


class TestPinExampleHackAutoFixConsent:
    """_try_auto_fix_import_error in pin_example_hack.py should also ask before pip install."""

    def test_pip_install_requires_user_confirmation(self, import_error_setup):
        """Same consent requirement for the duplicate in pin_example_hack.py."""
        code_file, example_file, error_output = import_error_setup

        from pdd.pin_example_hack import _try_auto_fix_import_error

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch('pdd.pin_example_hack.subprocess.run', return_value=mock_result) as mock_run, \
             patch('pdd.pin_example_hack.click') as mock_click:
            mock_click.confirm.return_value = True
            fixed, msg = _try_auto_fix_import_error(error_output, code_file, example_file)

        # FAILS on buggy code: click.confirm is never called
        mock_click.confirm.assert_called_once()

    def test_function_accepts_force_parameter(self):
        """The duplicate should also accept a force parameter."""
        from pdd.pin_example_hack import _try_auto_fix_import_error
        import inspect

        sig = inspect.signature(_try_auto_fix_import_error)
        assert 'force' in sig.parameters, \
            "_try_auto_fix_import_error should accept a 'force' parameter"
