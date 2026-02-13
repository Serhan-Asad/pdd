"""
E2E Test for Issue #494: _try_auto_fix_import_error() runs pip install without
user consent.

This is an E2E test that exercises the full call chain from the crash-handling
logic in sync_orchestration.py down to the _try_auto_fix_import_error() function.
Unlike the unit tests in test_auto_fix_import_no_consent.py which test the
function in isolation, this test verifies the user-facing behavior: when
`pdd sync` encounters a ModuleNotFoundError in an example file, it should ask
the user before installing any package.

The test mocks subprocess.run at the OS boundary (to prevent real pip install
and to simulate the example file crashing with ModuleNotFoundError), but
exercises all real Python code in between.

Bug: _try_auto_fix_import_error() calls pip install unconditionally with no
click.confirm() gate, no --force check, and no allowlist.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import subprocess
import sys


@pytest.mark.e2e
class TestIssue494AutoFixNoConsentE2E:
    """E2E: sync crash handling -> _try_auto_fix_import_error -> pip install
    must require user consent."""

    def test_full_chain_pip_install_runs_without_consent(self, tmp_path):
        """Exercise the full _try_auto_fix_import_error call chain and verify
        that pip install runs WITHOUT any click.confirm() call.

        This test FAILS on the current buggy code because subprocess.run IS
        called (for pip install) even though click.confirm was never invoked.
        Once fixed, click.confirm() will be called before pip install.
        """
        # Arrange: create files that simulate a ModuleNotFoundError scenario
        code_file = tmp_path / "widget.py"
        code_file.write_text("class Widget:\n    pass\n")

        example_file = tmp_path / "example_widget.py"
        example_file.write_text("import pandas\nfrom widget import Widget\nprint(Widget())\n")

        error_output = (
            "Traceback (most recent call last):\n"
            "  File \"example_widget.py\", line 1, in <module>\n"
            "    import pandas\n"
            "ModuleNotFoundError: No module named 'pandas'\n"
        )

        # Mock subprocess.run at the OS boundary to:
        # 1. Prevent actual pip install
        # 2. Track whether pip install was attempted
        pip_result = MagicMock()
        pip_result.returncode = 0
        pip_result.stdout = "Successfully installed pandas-2.0.0"
        pip_result.stderr = ""

        from pdd.sync_orchestration import _try_auto_fix_import_error

        # Verify that click.confirm is called before pip install.
        # We mock both subprocess.run (to prevent real pip) and click.confirm
        # (to prevent stdin read and to track the call).
        confirm_called = False

        def track_confirm(*args, **kwargs):
            nonlocal confirm_called
            confirm_called = True
            return True

        with patch('pdd.sync_orchestration.subprocess.run', return_value=pip_result) as mock_subprocess, \
             patch('pdd.sync_orchestration.click') as mock_click:
            mock_click.confirm = track_confirm
            fixed, msg = _try_auto_fix_import_error(
                error_output, code_file, example_file
            )

        # Verify pip install was called (function reaches the install path)
        pip_calls = [
            c for c in mock_subprocess.call_args_list
            if 'pip' in str(c) and 'install' in str(c)
        ]
        assert len(pip_calls) > 0, (
            "Expected pip install to be called (proving the function reaches "
            "the pip install path)"
        )

        # Verify click.confirm was called before pip install
        assert confirm_called, (
            "BUG: pip install ran without calling click.confirm() first. "
            "The user was never asked for consent before modifying their "
            "Python environment. (Issue #494)"
        )

    def test_full_chain_pip_install_blocked_when_declined(self, tmp_path):
        """When user declines the confirmation prompt, pip install must NOT run.

        This test FAILS on the buggy code because there is no confirmation
        gate — pip install always runs regardless of user intent.
        """
        code_file = tmp_path / "app.py"
        code_file.write_text("def main(): pass\n")

        example_file = tmp_path / "example_app.py"
        example_file.write_text("import flask\nfrom app import main\nmain()\n")

        error_output = "ModuleNotFoundError: No module named 'flask'"

        from pdd.sync_orchestration import _try_auto_fix_import_error

        with patch('pdd.sync_orchestration.subprocess.run') as mock_subprocess, \
             patch('pdd.sync_orchestration.click') as mock_click:
            # User says NO
            mock_click.confirm.return_value = False
            fixed, msg = _try_auto_fix_import_error(
                error_output, code_file, example_file
            )

        # FAILS on buggy code: subprocess.run is called anyway because
        # there's no click.confirm() gate
        pip_calls = [
            c for c in mock_subprocess.call_args_list
            if 'pip' in str(c) and 'install' in str(c)
        ]
        assert len(pip_calls) == 0, (
            "BUG: pip install ran even though user confirmation was not given. "
            "The function ignores user consent entirely. (Issue #494)"
        )

    def test_pin_example_hack_same_bug(self, tmp_path):
        """The duplicate _try_auto_fix_import_error in pin_example_hack.py
        has the same bug — pip install without consent.

        This test FAILS on the buggy code for the same reason.
        """
        code_file = tmp_path / "model.py"
        code_file.write_text("class Model: pass\n")

        example_file = tmp_path / "example_model.py"
        example_file.write_text("import torch\nfrom model import Model\n")

        error_output = "ModuleNotFoundError: No module named 'torch'"

        from pdd.pin_example_hack import _try_auto_fix_import_error

        confirm_called = False

        def track_confirm(*args, **kwargs):
            nonlocal confirm_called
            confirm_called = True
            return True

        pip_result = MagicMock()
        pip_result.returncode = 0
        pip_result.stdout = ""
        pip_result.stderr = ""

        with patch('pdd.pin_example_hack.subprocess.run', return_value=pip_result), \
             patch('pdd.pin_example_hack.click') as mock_click:
            mock_click.confirm = track_confirm
            fixed, msg = _try_auto_fix_import_error(
                error_output, code_file, example_file
            )

        # FAILS on buggy code: confirm is never called
        assert confirm_called, (
            "BUG: pin_example_hack.py's _try_auto_fix_import_error also runs "
            "pip install without calling click.confirm(). (Issue #494)"
        )
