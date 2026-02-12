"""
E2E Test for Issue #132: --quiet flag does not suppress non-essential output

Bug: The --quiet flag is stored in Click context but never propagated to
output-producing modules. Specifically:
1. preprocess() prints Rich Panels unconditionally (no quiet parameter)
2. llm_invoke.py logger is hardcoded to INFO level, ignoring --quiet

These tests exercise the FULL CLI path via CliRunner to verify that
--quiet suppresses non-essential output at the system level.
They FAIL on the current buggy code and PASS once the bug is fixed.
"""

import logging
import os
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

import pdd


@pytest.fixture(autouse=True)
def set_pdd_path(monkeypatch):
    """Set PDD_PATH to the pdd package directory for all tests."""
    pdd_package_dir = Path(pdd.__file__).parent
    monkeypatch.setenv("PDD_PATH", str(pdd_package_dir))


@pytest.mark.e2e
class TestIssue132QuietFlagE2E:
    """E2E tests: --quiet flag should suppress non-essential output via full CLI path."""

    def test_preprocess_quiet_suppresses_panels(self, tmp_path):
        """E2E: `pdd --quiet preprocess` should NOT show Rich Panel output.

        The bug: preprocess() prints "Starting prompt preprocessing" and
        "Preprocessing complete" panels unconditionally, ignoring --quiet.

        This test invokes the full CLI path and checks that panel text
        does NOT appear in output when --quiet is passed.

        FAILS on buggy code (panels still printed).
        PASSES once --quiet is propagated to preprocess().
        """
        from pdd.cli import cli

        # Create a properly named prompt file (language must be detectable)
        prompt_file = tmp_path / "test_python.prompt"
        prompt_file.write_text("Write a hello world function in Python.")

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(cli, ["--quiet", "preprocess", str(prompt_file)])

        combined = result.output + (result.stderr or "")

        # Bug assertion: these panel messages should NOT appear with --quiet
        assert "Starting prompt preprocessing" not in combined, (
            f"Bug #132 E2E: `pdd --quiet preprocess` still shows "
            f"'Starting prompt preprocessing' Rich Panel.\n"
            f"The --quiet flag is not propagated to preprocess().\n"
            f"Full output:\n{combined}"
        )
        assert "Preprocessing complete" not in combined, (
            f"Bug #132 E2E: `pdd --quiet preprocess` still shows "
            f"'Preprocessing complete' Rich Panel.\n"
            f"Full output:\n{combined}"
        )

    def test_preprocess_without_quiet_shows_panels(self, tmp_path):
        """Baseline: `pdd preprocess` (no --quiet) SHOULD show panel output.

        This confirms the panels exist in normal mode, so the quiet test
        is meaningful. This test should PASS on both buggy and fixed code.
        """
        from pdd.cli import cli

        prompt_file = tmp_path / "test_python.prompt"
        prompt_file.write_text("Write a hello world function in Python.")

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(cli, ["preprocess", str(prompt_file)])

        combined = result.output + (result.stderr or "")

        assert "Starting prompt preprocessing" in combined, (
            f"Expected 'Starting prompt preprocessing' panel in normal mode.\n"
            f"Full output:\n{combined}"
        )

    def test_llm_invoke_logger_respects_quiet(self):
        """E2E: The pdd.llm_invoke logger should suppress INFO when --quiet is active.

        The bug: llm_invoke.py hardcodes logger level to INFO with no
        mechanism to raise it when --quiet is passed via Click context.

        We verify by checking that the logger level is set to INFO
        (the bug) rather than WARNING+ (the fix).

        FAILS on buggy code (logger hardcoded to INFO).
        PASSES once --quiet propagation raises the log level.
        """
        import importlib
        import pdd.llm_invoke

        importlib.reload(pdd.llm_invoke)
        pdd_logger = logging.getLogger("pdd.llm_invoke")

        # Bug: logger is explicitly set to INFO, never raised for quiet mode
        assert pdd_logger.level > logging.INFO, (
            f"Bug #132 E2E: pdd.llm_invoke logger level is "
            f"{logging.getLevelName(pdd_logger.level)} ({pdd_logger.level}). "
            f"It should be WARNING or higher when quiet mode is active, "
            f"but the logger is hardcoded to INFO with no quiet-awareness."
        )

    def test_quiet_mode_preserves_errors(self, tmp_path):
        """E2E: --quiet should still show errors (not suppress everything).

        This is a correctness baseline: errors must remain visible.
        PASSES on both buggy and fixed code.
        """
        from pdd.cli import cli

        # Use a nonexistent file to trigger an error
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(cli, ["--quiet", "preprocess", "/nonexistent/file.prompt"])

        # Should exit with error (Click catches bad path)
        assert result.exit_code != 0, (
            "Expected non-zero exit code for nonexistent file"
        )
