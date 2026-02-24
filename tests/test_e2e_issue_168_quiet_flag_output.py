"""
E2E CLI Test for Issue #168: --quiet flag does not suppress output

Tests the full CLI path (through pdd.cli:cli entry point) to verify that
the --quiet flag suppresses non-essential output from preprocess() during
code generation.

Unlike the unit tests in tests/test_preprocess.py which test preprocess()
in isolation (and fail with TypeError since it lacks a quiet parameter),
these E2E tests invoke the full CLI entry point and exercise the real
command dispatch, flag propagation, and output pipeline.

The bug: `pdd --quiet generate prompts/test.prompt` still displays:
- Rich panels ("Starting prompt preprocessing", "Preprocessing complete")
- "Doubling curly brackets..." progress messages

Root cause: preprocess() has no quiet parameter, and code_generator_main
never passes the quiet flag to preprocess(). The --quiet flag is correctly
parsed in cli.py and stored in ctx.obj["quiet"], but it is never propagated
to the preprocess() function.

See: https://github.com/Serhan-Asad/pdd/issues/168
"""

import pytest
from click.testing import CliRunner
from unittest.mock import patch


pytestmark = pytest.mark.e2e


class TestIssue168QuietFlagE2E:
    """E2E tests for Issue #168: Full CLI path for --quiet flag output suppression."""

    def test_full_cli_quiet_generate_suppresses_preprocess_output(self, tmp_path):
        """E2E: Invoke `pdd --quiet --local generate <file>` through the full CLI
        entry point and verify that preprocess() output is suppressed.

        This exercises the real CLI dispatch path:
        pdd.cli:cli → generate command → code_generator_main → pdd_preprocess()

        The mock replaces code_generator_main with a function that calls the
        real pdd_preprocess() on the prompt content. This simulates the actual
        code path without requiring LLM API calls.

        Bug: preprocess() has no quiet parameter, so its console.print() calls
        produce Rich panels and progress messages even when --quiet is set.
        code_generator_main never passes quiet to preprocess().

        This test FAILS on the current buggy code, PASSES after the fix.
        """
        # Create a real prompt file with content that triggers double_curly
        prompt_file = tmp_path / "test.prompt"
        prompt_file.write_text("Generate a {hello} world function in Python")

        from pdd.preprocess import preprocess as real_preprocess

        def mock_code_gen_main(ctx, prompt_file, output, **kwargs):
            """Mock code_generator_main that calls real pdd_preprocess().

            This simulates the actual code path in code_generator_main.py
            (lines 904-906) where pdd_preprocess is called on the prompt
            content. By calling the real preprocess(), we exercise the
            console.print() calls that should be suppressed by --quiet.
            """
            with open(prompt_file, 'r') as f:
                content = f.read()
            # This is what the real code_generator_main does (lines 904-906)
            real_preprocess(content, recursive=True, double_curly_brackets=True)
            return ("generated code", False, 0.0, "test-model")

        with patch("pdd.core.cli.auto_update"), \
             patch("pdd.commands.generate.code_generator_main", side_effect=mock_code_gen_main):
            from pdd.cli import cli

            runner = CliRunner(mix_stderr=False)
            result = runner.invoke(
                cli,
                ["--quiet", "--local", "generate", str(prompt_file)],
            )

        # Combine stdout and stderr for full output check
        full_output = result.output + (result.stderr or "")

        # BUG ASSERTION 1: Rich panel "Starting prompt preprocessing" should NOT appear
        assert "Starting prompt preprocessing" not in full_output, (
            f"Bug #168 E2E: The 'Starting prompt preprocessing' Rich panel "
            f"was displayed despite the --quiet flag being set.\n"
            f"This proves that preprocess() does not respect quiet mode.\n"
            f"Full output:\n{full_output}"
        )

        # BUG ASSERTION 2: Progress message "Doubling curly brackets..." should NOT appear
        assert "Doubling curly brackets" not in full_output, (
            f"Bug #168 E2E: The 'Doubling curly brackets...' progress message "
            f"was displayed despite the --quiet flag being set.\n"
            f"This proves that double_curly() does not respect quiet mode.\n"
            f"Full output:\n{full_output}"
        )

        # BUG ASSERTION 3: Rich panel "Preprocessing complete" should NOT appear
        assert "Preprocessing complete" not in full_output, (
            f"Bug #168 E2E: The 'Preprocessing complete' Rich panel "
            f"was displayed despite the --quiet flag being set.\n"
            f"This proves that preprocess() does not respect quiet mode.\n"
            f"Full output:\n{full_output}"
        )

    def test_full_cli_non_quiet_generate_shows_preprocess_output(self, tmp_path):
        """E2E regression guard: Without --quiet, preprocess output should appear.

        This test verifies that the normal (non-quiet) behavior continues
        to work correctly. It should PASS both before and after the fix.
        """
        # Create a real prompt file
        prompt_file = tmp_path / "test.prompt"
        prompt_file.write_text("Generate a {hello} world function in Python")

        from pdd.preprocess import preprocess as real_preprocess

        def mock_code_gen_main(ctx, prompt_file, output, **kwargs):
            """Mock code_generator_main that calls real pdd_preprocess()."""
            with open(prompt_file, 'r') as f:
                content = f.read()
            real_preprocess(content, recursive=True, double_curly_brackets=True)
            return ("generated code", False, 0.0, "test-model")

        with patch("pdd.core.cli.auto_update"), \
             patch("pdd.commands.generate.code_generator_main", side_effect=mock_code_gen_main):
            from pdd.cli import cli

            runner = CliRunner(mix_stderr=False)
            result = runner.invoke(
                cli,
                ["--local", "generate", str(prompt_file)],
            )

        full_output = result.output + (result.stderr or "")

        # Without --quiet, preprocess output SHOULD appear
        has_preprocess_output = (
            "Starting prompt preprocessing" in full_output or
            "Doubling curly brackets" in full_output or
            "Preprocessing complete" in full_output
        )
        assert has_preprocess_output, (
            f"Regression guard: preprocess output should appear without --quiet.\n"
            f"Full output:\n{full_output}"
        )

    def test_full_cli_quiet_flag_reaches_ctx_obj(self, tmp_path):
        """E2E: Verify --quiet flag is correctly propagated to ctx.obj.

        This test verifies the first part of the chain works: the CLI
        correctly sets ctx.obj["quiet"] = True when --quiet is passed.
        The bug is in the SECOND part: ctx.obj["quiet"] is never forwarded
        to preprocess() or load_prompt_template().

        This test should PASS on both buggy and fixed code.
        """
        prompt_file = tmp_path / "test.prompt"
        prompt_file.write_text("Simple test prompt")

        captured_quiet = {}

        def mock_code_gen_main(ctx, prompt_file, output, **kwargs):
            """Capture the quiet flag from ctx.obj."""
            captured_quiet["value"] = ctx.obj.get("quiet", False)
            return ("generated code", False, 0.0, "test-model")

        with patch("pdd.core.cli.auto_update"), \
             patch("pdd.commands.generate.code_generator_main", side_effect=mock_code_gen_main):
            from pdd.cli import cli

            runner = CliRunner(mix_stderr=False)
            result = runner.invoke(
                cli,
                ["--quiet", "--local", "generate", str(prompt_file)],
            )

        # The --quiet flag should be in ctx.obj
        assert captured_quiet.get("value") is True, (
            f"The --quiet flag was not propagated to ctx.obj['quiet'].\n"
            f"Expected True, got {captured_quiet.get('value')}\n"
            f"Exit code: {result.exit_code}\n"
            f"Output: {result.output}"
        )
