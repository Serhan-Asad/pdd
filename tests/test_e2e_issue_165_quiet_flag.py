"""
E2E CLI Test for Issue #165: --quiet flag does not suppress output.

Tests the full `pdd --quiet generate` CLI path (through pdd.cli:cli entry point)
to verify that --quiet suppresses all non-essential output including Rich Panels,
INFO log lines, success messages, and warning messages.

Unlike the unit tests in tests/test_quiet_flag_suppression.py which test individual
functions in isolation, these E2E tests invoke the full CLI entry point (`pdd.cli:cli`)
to exercise the real command dispatch and verify user-visible behavior.

Only the LLM network layer is mocked — all other code (preprocess, load_prompt_template,
code_generator_main, logging setup) runs for real.

Bug: The --quiet flag is stored in Click context at cli.py:333 but never propagated
to preprocess(), load_prompt_template(), code_generator(), or set_verbose_logging().
These functions lack a quiet parameter and emit Rich Panels, success messages, and
INFO log lines unconditionally.

See: https://github.com/Serhan-Asad/pdd/issues/165
"""

import os
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


@pytest.mark.e2e
class TestIssue165QuietFlagE2E:
    """E2E tests for Issue #165: --quiet flag should suppress non-essential output."""

    def _create_prompt_file(self, tmp_path: Path) -> Path:
        """Create a minimal .prompt file for testing."""
        prompt_file = tmp_path / "test_quiet.prompt"
        prompt_file.write_text("% Write a hello world function in Python.\n")
        return prompt_file

    def test_quiet_flag_suppresses_preprocessing_panels(self, tmp_path):
        """E2E: `pdd --quiet generate <prompt>` should not show Rich Panels.

        User scenario:
        1. User runs `pdd --quiet generate prompts/hello.prompt`
        2. Expected: No Rich Panels ("Starting prompt preprocessing", "Preprocessing complete")
        3. Actual (buggy): Both panels are printed despite --quiet

        This exercises the full CLI dispatch path:
        pdd.cli:cli -> generate command -> code_generator_main -> preprocess
        """
        from click.testing import CliRunner
        from pdd.cli import cli

        prompt_file = self._create_prompt_file(tmp_path)

        # Mock only the LLM call — everything else runs for real
        with patch("pdd.code_generator.llm_invoke") as mock_llm, \
             patch("pdd.code_generator.unfinished_prompt") as mock_unfinished, \
             patch("pdd.code_generator.postprocess") as mock_postprocess, \
             patch("pdd.code_generator_main.construct_paths") as mock_cp:

            mock_cp.return_value = (
                {},  # config
                {"prompt_file": "% Write a hello world function"},  # input_files
                {"output": str(tmp_path / "output.py")},  # output_paths
                "python",  # language
            )
            mock_llm.return_value = {
                "result": "def hello(): print('hello')",
                "cost": 0.001,
                "model_name": "test-model",
            }
            mock_unfinished.return_value = ("", True, 0.0, "")
            mock_postprocess.return_value = ("def hello(): print('hello')", 0.001, "test-model")

            runner = CliRunner(mix_stderr=False)
            result = runner.invoke(cli, [
                "--quiet",
                "--force",
                "--local",
                "generate",
                str(prompt_file),
            ])

        # BUG ASSERTION 1: Rich Panel "Starting prompt preprocessing" should be suppressed
        assert "Starting prompt preprocessing" not in result.output, (
            f"Bug #165 E2E: The full CLI path with --quiet still shows the "
            f"'Starting prompt preprocessing' Rich Panel.\n"
            f"The --quiet flag is stored in ctx.obj but never passed to preprocess().\n"
            f"Full output:\n{result.output}"
        )

        # BUG ASSERTION 2: Rich Panel "Preprocessing complete" should be suppressed
        assert "Preprocessing complete" not in result.output, (
            f"Bug #165 E2E: The full CLI path with --quiet still shows the "
            f"'Preprocessing complete' Rich Panel.\n"
            f"Full output:\n{result.output}"
        )

    def test_quiet_flag_suppresses_doubling_curly_message(self, tmp_path):
        """E2E: `pdd --quiet generate <prompt>` should not show 'Doubling curly brackets'.

        User scenario:
        1. User runs `pdd --quiet generate prompts/hello.prompt`
        2. Expected: No "Doubling curly brackets..." status message
        3. Actual (buggy): Message is printed because preprocess() has no quiet param

        This exercises a different output path from test 1 — the double_curly()
        function emits its own message independently of the Rich Panels.
        """
        from click.testing import CliRunner
        from pdd.cli import cli

        # Use a prompt with curly braces to trigger the "Doubling curly brackets" path
        prompt_file = tmp_path / "test_curly.prompt"
        prompt_file.write_text("% Write code using {variables} and {{templates}}.\n")

        with patch("pdd.code_generator.llm_invoke") as mock_llm, \
             patch("pdd.code_generator.unfinished_prompt") as mock_unfinished, \
             patch("pdd.code_generator.postprocess") as mock_postprocess, \
             patch("pdd.code_generator_main.construct_paths") as mock_cp:

            mock_cp.return_value = (
                {},
                {"prompt_file": "% Write code using {variables}"},
                {"output": str(tmp_path / "output.py")},
                "python",
            )
            mock_llm.return_value = {
                "result": "def hello(): print('hello')",
                "cost": 0.001,
                "model_name": "test-model",
            }
            mock_unfinished.return_value = ("", True, 0.0, "")
            mock_postprocess.return_value = ("def hello(): print('hello')", 0.001, "test-model")

            runner = CliRunner(mix_stderr=False)
            result = runner.invoke(cli, [
                "--quiet",
                "--force",
                "--local",
                "generate",
                str(prompt_file),
            ])

        # BUG ASSERTION: "Doubling curly brackets" should be suppressed
        assert "Doubling curly brackets" not in result.output, (
            f"Bug #165 E2E: The full CLI path with --quiet still shows the "
            f"'Doubling curly brackets...' message from preprocess().\n"
            f"preprocess() has no quiet parameter and prints unconditionally.\n"
            f"Full output:\n{result.output}"
        )

    def test_quiet_flag_combined_output_check(self, tmp_path):
        """E2E: `pdd --quiet generate <prompt>` overall output should be minimal.

        This is the primary E2E test that checks all suppressed outputs at once,
        matching the exact user experience described in issue #165.

        The user expects --quiet to suppress:
        - INFO log lines (pdd.llm_invoke - INFO - ...)
        - Rich Panels ("Starting prompt preprocessing", "Preprocessing complete")
        - "Doubling curly brackets..." messages
        - WARNING log lines (cloud fallback warnings)
        - "Successfully loaded prompt: ..." messages
        """
        from click.testing import CliRunner
        from pdd.cli import cli

        prompt_file = self._create_prompt_file(tmp_path)

        with patch("pdd.code_generator.llm_invoke") as mock_llm, \
             patch("pdd.code_generator.unfinished_prompt") as mock_unfinished, \
             patch("pdd.code_generator.postprocess") as mock_postprocess, \
             patch("pdd.code_generator_main.construct_paths") as mock_cp:

            mock_cp.return_value = (
                {},
                {"prompt_file": "% Write a hello world function"},
                {"output": str(tmp_path / "output.py")},
                "python",
            )
            mock_llm.return_value = {
                "result": "def hello(): print('hello')",
                "cost": 0.001,
                "model_name": "test-model",
            }
            mock_unfinished.return_value = ("", True, 0.0, "")
            mock_postprocess.return_value = ("def hello(): print('hello')", 0.001, "test-model")

            runner = CliRunner(mix_stderr=False)
            result = runner.invoke(cli, [
                "--quiet",
                "--force",
                "--local",
                "generate",
                str(prompt_file),
            ])

        combined_output = result.output + (result.stderr if hasattr(result, 'stderr') and result.stderr else "")

        # All non-essential outputs that should be suppressed by --quiet
        suppressed_patterns = [
            ("Starting prompt preprocessing", "Rich Panel from preprocess()"),
            ("Preprocessing complete", "Rich Panel from preprocess()"),
            ("Doubling curly brackets", "Status message from preprocess()"),
            ("Successfully loaded prompt", "Success message from load_prompt_template()"),
        ]

        violations = []
        for pattern, source in suppressed_patterns:
            if pattern in combined_output:
                violations.append(f"  - '{pattern}' ({source})")

        assert not violations, (
            f"Bug #165 E2E: `pdd --quiet generate` still emits {len(violations)} "
            f"non-essential output(s) that should be suppressed:\n"
            + "\n".join(violations)
            + f"\n\nFull output:\n{combined_output}"
        )


@pytest.mark.e2e
class TestIssue165QuietFlagSubprocess:
    """E2E tests using subprocess for the most realistic test of --quiet behavior.

    These tests spawn a real `python -m pdd.cli` process to exercise the exact
    code path a user takes when running `pdd --quiet generate`.
    """

    def test_subprocess_quiet_suppresses_output(self, tmp_path):
        """E2E (subprocess): `pdd --quiet generate <prompt>` should not emit panels.

        This is the most realistic E2E test — it spawns a real process with
        --quiet and checks that preprocessing panels do not appear in output.

        The test creates a wrapper script that mocks only the LLM layer
        (to avoid real API calls) while running everything else for real.
        """
        import subprocess
        import tempfile

        # Create a minimal prompt file
        prompt_file = tmp_path / "test_quiet_subprocess.prompt"
        prompt_file.write_text("% Write a hello world function in Python.\n")

        project_root = str(Path(__file__).resolve().parent.parent)

        # Wrapper script that mocks LLM calls but runs everything else for real
        wrapper_code = textwrap.dedent(f"""\
            import sys
            import os
            os.environ.setdefault('PDD_FORCE_LOCAL', '1')

            from unittest.mock import patch, MagicMock

            # Mock only the LLM network call
            with patch("pdd.code_generator.llm_invoke") as mock_llm, \\
                 patch("pdd.code_generator.unfinished_prompt") as mock_unfinished, \\
                 patch("pdd.code_generator.postprocess") as mock_postprocess, \\
                 patch("pdd.code_generator_main.construct_paths") as mock_cp:

                mock_cp.return_value = (
                    {{}},
                    {{"prompt_file": "% Write a hello world function"}},
                    {{"output": "{tmp_path / 'output.py'}"}},
                    "python",
                )
                mock_llm.return_value = {{
                    "result": "def hello(): print('hello')",
                    "cost": 0.001,
                    "model_name": "test-model",
                }}
                mock_unfinished.return_value = ("", True, 0.0, "")
                mock_postprocess.return_value = ("def hello(): print('hello')", 0.001, "test-model")

                from pdd.cli import cli
                sys.exit(cli([
                    "--quiet", "--force", "--local",
                    "generate", "{prompt_file}",
                ], standalone_mode=True))
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(wrapper_code)
            wrapper_path = f.name

        env = os.environ.copy()
        env["PYTHONPATH"] = project_root

        try:
            proc = subprocess.run(
                [sys.executable, wrapper_path],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=project_root,
                env=env,
            )

            full_output = proc.stdout + proc.stderr

            # BUG ASSERTION: These should NOT appear with --quiet
            assert "Starting prompt preprocessing" not in full_output, (
                f"Bug #165 E2E (subprocess): 'Starting prompt preprocessing' panel "
                f"appeared in output despite --quiet flag.\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
            assert "Preprocessing complete" not in full_output, (
                f"Bug #165 E2E (subprocess): 'Preprocessing complete' panel "
                f"appeared in output despite --quiet flag.\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        finally:
            os.unlink(wrapper_path)
