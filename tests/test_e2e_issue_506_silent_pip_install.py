"""
E2E regression tests for Issue #506: Auto pip install in _try_auto_fix_import_error()
is silent — needs confirmation + requirements update.

These tests exercise the full code path through _try_auto_fix_import_error() with
real file system artifacts (tmp_path code/example files) and a real error string,
verifying that the function asks for user confirmation before running pip install
and updates the project's dependency file afterward.

The bug: _try_auto_fix_import_error() silently runs `pip install <package>` with
capture_output=True, no click.confirm(), no terminal output, and no dependency
file update. The call site also doesn't pass `force`/`quiet` parameters.
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestIssue506SilentPipInstallE2E:
    """
    E2E tests verifying that _try_auto_fix_import_error() prompts the user
    before installing packages and updates dependency files.
    """

    def _make_project(self, tmp_path):
        """Create minimal project files: a code file, an example file, and error output."""
        code_file = tmp_path / "my_app.py"
        code_file.write_text("import requests\nprint('hello')\n")

        example_file = tmp_path / "example_my_app.py"
        example_file.write_text("from my_app import *\nprint('running')\n")

        error_output = "ModuleNotFoundError: No module named 'requests'"
        return code_file, example_file, error_output

    def test_pip_install_runs_without_any_confirmation(self):
        """
        E2E: Call _try_auto_fix_import_error() with an external package
        ModuleNotFoundError and verify that it does NOT call click.confirm()
        before running pip install. This is the core bug.

        The fix should add a click.confirm() call. Once fixed, this test
        passes because click.confirm will be called.
        """
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            code_file, example_file, error_output = self._make_project(tmp_path)

            # Patch subprocess.run so we don't actually pip install,
            # but do NOT patch click.confirm — we want to see if it's called.
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""

            with patch("pdd.sync_orchestration.subprocess.run", return_value=mock_result) as mock_run, \
                 patch("pdd.sync_orchestration.click.confirm") as mock_confirm:

                from pdd.sync_orchestration import _try_auto_fix_import_error
                fixed, msg = _try_auto_fix_import_error(error_output, code_file, example_file)

            # BUG: The current code never calls click.confirm before pip install.
            # The fix should call click.confirm() to ask the user.
            assert mock_confirm.called, (
                "BUG: _try_auto_fix_import_error() ran pip install without calling "
                "click.confirm() first. Users should be prompted before packages are "
                "installed into their environment."
            )

    def test_no_dependency_file_update_after_install(self):
        """
        E2E: After a successful pip install, verify that the function does NOT
        update any dependency file (requirements.txt). This confirms the bug —
        the fix should append the package to requirements.txt.
        """
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            code_file, example_file, error_output = self._make_project(tmp_path)

            # Create a requirements.txt in the "project root"
            req_file = tmp_path / "requirements.txt"
            req_file.write_text("flask\n")

            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""

            with patch("pdd.sync_orchestration.subprocess.run", return_value=mock_result), \
                 patch("pdd.sync_orchestration.click.confirm", return_value=True, create=True):
                from pdd.sync_orchestration import _try_auto_fix_import_error
                fixed, msg = _try_auto_fix_import_error(error_output, code_file, example_file)

            assert fixed, f"Expected auto-fix to succeed, got: {msg}"

            # BUG: The installed package is never added to requirements.txt.
            contents = req_file.read_text()
            assert "requests" in contents, (
                "BUG: After pip installing 'requests', the package was not added to "
                f"requirements.txt. File contents: {contents!r}"
            )

    def test_call_site_does_not_pass_force_quiet(self):
        """
        E2E: Verify that the call site in _run_sync_operations passes force/quiet
        to _try_auto_fix_import_error. We inspect the function signature to confirm
        it accepts these parameters.

        The current buggy code's function signature is:
            _try_auto_fix_import_error(error_output, code_file, example_file)
        The fix should add force and quiet parameters.
        """
        import inspect
        from pdd.sync_orchestration import _try_auto_fix_import_error

        sig = inspect.signature(_try_auto_fix_import_error)
        param_names = list(sig.parameters.keys())

        # BUG: The function doesn't accept force/quiet parameters
        assert "force" in param_names, (
            f"BUG: _try_auto_fix_import_error() does not accept a 'force' parameter. "
            f"Current params: {param_names}. The --force flag should skip confirmation."
        )
        assert "quiet" in param_names, (
            f"BUG: _try_auto_fix_import_error() does not accept a 'quiet' parameter. "
            f"Current params: {param_names}. The --quiet flag should suppress output."
        )
