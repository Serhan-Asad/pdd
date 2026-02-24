"""
E2E tests for GitHub Issue #166: --quiet flag does not suppress output.

These tests exercise the full system path from CLI invocation through
code_generator_main to preprocess(), verifying that the --quiet flag
fails to suppress non-essential output at the system level.

Unlike the unit tests in test_quiet_flag_issue_166.py which test individual
functions in isolation, these E2E tests invoke the real CLI and let the
real preprocess()/double_curly() run unmocked, demonstrating that the
quiet flag is never threaded through to suppress their output.

All tests should FAIL on the current buggy code because:
- preprocess() has no quiet parameter and prints Rich panels unconditionally
- double_curly() prints "Doubling curly brackets..." unconditionally
- code_generator_main() never passes quiet to pdd_preprocess()
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from click.testing import CliRunner


@pytest.fixture(autouse=True)
def set_pdd_path(monkeypatch):
    """Set PDD_PATH to the pdd package directory for all tests in this module."""
    import pdd
    pdd_package_dir = Path(pdd.__file__).parent
    monkeypatch.setenv("PDD_PATH", str(pdd_package_dir))


class TestE2EIssue166QuietFlagCLI:
    """E2E tests: CLI --quiet flag should suppress preprocess output.

    These tests use CliRunner to invoke the real CLI with --quiet, while
    replacing the preprocess module's console with a mock to track what
    output would have been produced.

    Note: CliRunner does not capture Rich Console output in result.output.
    We replace pdd.preprocess.console with a mock to detect leaked output.
    """

    def test_e2e_cli_quiet_generate_suppresses_preprocess_panels(self, tmp_path):
        """
        E2E: `pdd --quiet --local generate <file>` should not produce
        Rich panels from preprocess().

        Exercises: CLI -> generate -> code_generator_main -> pdd_preprocess()
        Only LLM and construct_paths are mocked; preprocess logic runs for real
        (but console is mocked to capture what it would print).

        FAILS on buggy code because preprocess() has no quiet parameter and
        unconditionally calls console.print(Panel(...)).
        """
        from pdd.cli import cli

        prompt_file = tmp_path / "test_quiet.prompt"
        prompt_file.write_text("Generate a hello world function in Python")

        runner = CliRunner(mix_stderr=False)
        mock_config = {"language": "python", "tests_dir": str(tmp_path)}
        mock_output_paths = {"output": str(tmp_path / "output.py")}

        with patch("pdd.code_generator_main.construct_paths") as mock_cp, \
             patch("pdd.code_generator_main.local_code_generator_func") as mock_llm, \
             patch("pdd.core.cli.auto_update"), \
             patch("pdd.core.dump.garbage_collect_core_dumps"), \
             patch("pdd.preprocess.console") as mock_preprocess_console:

            mock_cp.return_value = (mock_config, {}, mock_output_paths, "python")
            mock_llm.return_value = ("print('hello world')", 0.01, "test-model")

            result = runner.invoke(cli, [
                "--quiet", "--no-core-dump", "--local",
                "generate", str(prompt_file)
            ])

        # Sanity: preprocess was called (console.print was invoked)
        assert mock_preprocess_console.print.call_count > 0, (
            "Sanity check: preprocess console.print was never called. "
            f"Exit code: {result.exit_code}, Output: {result.output}"
        )

        # BUG ASSERTION: In quiet mode, no Rich Panels should be printed
        panel_messages = []
        for c in mock_preprocess_console.print.call_args_list:
            for arg in c.args:
                if hasattr(arg, "renderable"):
                    panel_messages.append(str(arg.renderable))

        assert not panel_messages, (
            "BUG (Issue #166): Rich Panels printed by preprocess() during "
            "`pdd --quiet generate` despite --quiet flag.\n"
            f"Panels found: {panel_messages}"
        )

    def test_e2e_cli_quiet_generate_suppresses_doubling_message(self, tmp_path):
        """
        E2E: `pdd --quiet --local generate <file>` should not produce
        "Doubling curly brackets..." message from double_curly().

        Exercises: CLI -> generate -> code_generator_main -> pdd_preprocess()
                   -> double_curly()

        FAILS on buggy code because double_curly() has no quiet parameter
        and unconditionally calls console.print("Doubling curly brackets...").
        """
        from pdd.cli import cli

        prompt_file = tmp_path / "test_quiet_doubling.prompt"
        prompt_file.write_text("Generate a function that says {greeting}")

        runner = CliRunner(mix_stderr=False)
        mock_config = {"language": "python", "tests_dir": str(tmp_path)}
        mock_output_paths = {"output": str(tmp_path / "output.py")}

        with patch("pdd.code_generator_main.construct_paths") as mock_cp, \
             patch("pdd.code_generator_main.local_code_generator_func") as mock_llm, \
             patch("pdd.core.cli.auto_update"), \
             patch("pdd.core.dump.garbage_collect_core_dumps"), \
             patch("pdd.preprocess.console") as mock_preprocess_console:

            mock_cp.return_value = (mock_config, {}, mock_output_paths, "python")
            mock_llm.return_value = ("def greet(): pass", 0.01, "test-model")

            result = runner.invoke(cli, [
                "--quiet", "--no-core-dump", "--local",
                "generate", str(prompt_file)
            ])

        # BUG ASSERTION: "Doubling curly brackets..." should not appear
        text_messages = []
        for c in mock_preprocess_console.print.call_args_list:
            for arg in c.args:
                if isinstance(arg, str):
                    text_messages.append(arg)

        assert not any("Doubling curly brackets" in m for m in text_messages), (
            "BUG (Issue #166): 'Doubling curly brackets...' printed during "
            "`pdd --quiet generate` despite --quiet flag.\n"
            f"Messages found: {text_messages}"
        )


class TestE2EIssue166QuietFlagIntegration:
    """Integration tests: code_generator_main does not thread quiet to preprocess."""

    def test_e2e_code_generator_main_leaks_preprocess_output_in_quiet_mode(
        self, tmp_path
    ):
        """
        Integration: code_generator_main(quiet=True) still produces preprocess output
        because it never passes quiet to pdd_preprocess().

        Calls code_generator_main directly with a Click context where quiet=True.
        Only construct_paths and the LLM call are mocked; preprocess runs for real.

        FAILS on buggy code because code_generator_main reads quiet from ctx.obj
        but never passes it to pdd_preprocess() (lines 904-906).
        """
        import click
        from pdd.code_generator_main import code_generator_main
        import pdd.preprocess as preprocess_mod

        prompt_file = tmp_path / "test_integration.prompt"
        prompt_file.write_text("Write a greeting function")

        ctx = click.Context(click.Command("generate"))
        ctx.ensure_object(dict)
        ctx.obj["quiet"] = True
        ctx.obj["verbose"] = False
        ctx.obj["force"] = True
        ctx.obj["local"] = True
        ctx.obj["strength"] = 0.75
        ctx.obj["temperature"] = 0.0
        ctx.obj["time"] = 0.5
        ctx.obj["context"] = None

        mock_config = {"language": "python", "tests_dir": str(tmp_path)}
        mock_output_paths = {"output": str(tmp_path / "output.py")}

        preprocess_prints = []
        original_print = preprocess_mod.console.print

        def capturing_print(*args, **kwargs):
            preprocess_prints.append(args)
            return original_print(*args, **kwargs)

        with patch("pdd.code_generator_main.construct_paths") as mock_cp, \
             patch("pdd.code_generator_main.local_code_generator_func") as mock_llm, \
             patch.object(preprocess_mod.console, "print", side_effect=capturing_print):

            mock_cp.return_value = (mock_config, {}, mock_output_paths, "python")
            mock_llm.return_value = ("print('hello')", 0.01, "test-model")

            result = code_generator_main(
                ctx=ctx,
                prompt_file=str(prompt_file),
                output=None,
                original_prompt_file_path=None,
                force_incremental_flag=False,
            )

        # Sanity: preprocess was actually called
        assert len(preprocess_prints) > 0, (
            "Sanity check failed: preprocess console.print was never called"
        )

        # BUG ASSERTION: In quiet mode, preprocess should not have printed
        # Panel objects or "Doubling curly brackets..." messages.
        panel_messages = []
        text_messages = []
        for call_args in preprocess_prints:
            for arg in call_args:
                if hasattr(arg, "renderable"):
                    panel_messages.append(str(arg.renderable))
                elif isinstance(arg, str):
                    text_messages.append(arg)

        assert not panel_messages, (
            "BUG (Issue #166): Rich Panels printed by preprocess() despite "
            f"quiet=True in context.\n"
            f"Panels found: {panel_messages}"
        )
        assert not any("Doubling curly brackets" in m for m in text_messages), (
            "BUG (Issue #166): 'Doubling curly brackets...' printed by "
            f"preprocess() despite quiet=True.\n"
            f"Messages found: {text_messages}"
        )

    def test_e2e_preprocess_rejects_quiet_parameter(self):
        """
        Integration: preprocess() cannot accept a quiet parameter, making it
        impossible for the CLI's --quiet flag to suppress its output.

        This directly demonstrates the root cause: the function signature
        lacks the quiet parameter entirely.

        FAILS on buggy code with TypeError because preprocess() doesn't
        accept quiet=True. After the fix, this test will pass.
        """
        from pdd.preprocess import preprocess

        printed_items = []

        with patch("pdd.preprocess.console") as mock_console:
            mock_console.print.side_effect = (
                lambda *args, **kwargs: printed_items.extend(args)
            )

            # This should work after the fix (quiet=True suppresses output)
            # Currently raises TypeError: unexpected keyword argument 'quiet'
            result = preprocess(
                "Hello world",
                recursive=False,
                double_curly_brackets=True,
                quiet=True,
            )

        assert result == "Hello world"

        # After fix: verify quiet=True actually suppressed panel output
        panel_items = [
            item for item in printed_items if hasattr(item, "renderable")
        ]
        assert not panel_items, (
            "BUG (Issue #166): preprocess(quiet=True) should suppress panels "
            f"but produced: {[str(p.renderable) for p in panel_items]}"
        )

    def test_e2e_full_generate_path_quiet_no_leaked_output(self, tmp_path):
        """
        E2E: Full CLI generate path with --quiet should produce zero
        non-essential preprocess output.

        This is the most comprehensive test — it invokes the real CLI with
        --quiet, tracks ALL preprocess console.print calls via mock, and
        asserts that none of them produce panels or informational messages.

        FAILS on buggy code because preprocess(), double_curly(), and
        other functions produce output regardless of quiet mode.
        """
        from pdd.cli import cli

        prompt_file = tmp_path / "test_full_quiet.prompt"
        prompt_file.write_text("Create a simple calculator with {operator} support")

        runner = CliRunner(mix_stderr=False)
        mock_config = {"language": "python", "tests_dir": str(tmp_path)}
        mock_output_paths = {"output": str(tmp_path / "output.py")}

        with patch("pdd.code_generator_main.construct_paths") as mock_cp, \
             patch("pdd.code_generator_main.local_code_generator_func") as mock_llm, \
             patch("pdd.core.cli.auto_update"), \
             patch("pdd.core.dump.garbage_collect_core_dumps"), \
             patch("pdd.preprocess.console") as mock_preprocess_console:

            mock_cp.return_value = (mock_config, {}, mock_output_paths, "python")
            mock_llm.return_value = ("def calc(): pass", 0.01, "test-model")

            result = runner.invoke(cli, [
                "--quiet", "--no-core-dump", "--local",
                "generate", str(prompt_file)
            ])

        # Collect ALL output that should have been suppressed
        leaked_panels = []
        leaked_messages = []
        for c in mock_preprocess_console.print.call_args_list:
            for arg in c.args:
                if hasattr(arg, "renderable"):
                    leaked_panels.append(str(arg.renderable))
                elif isinstance(arg, str) and arg.strip():
                    leaked_messages.append(arg)

        all_leaked = leaked_panels + leaked_messages
        assert not all_leaked, (
            "BUG (Issue #166): --quiet flag did not suppress preprocess output.\n"
            f"Leaked panels: {leaked_panels}\n"
            f"Leaked messages: {leaked_messages}"
        )
