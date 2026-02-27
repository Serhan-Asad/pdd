"""
E2E Test for Issue #625: pdd generate calls its own functions with wrong arguments.

This test exercises the full code generation pipeline to verify that when the LLM
generates Python code with mismatched function arguments (e.g., calling a 4-parameter
function with 3 arguments), the post-generation pipeline detects and reports the issue.

Bug: The code_generator pipeline has zero post-generation validation for function
argument consistency. The LLM can generate code where fetch_file_content(owner, repo,
path, headers) is called as fetch_file_content(repo_url, 'README.md', github_token),
and the pipeline writes this to disk without any warning.

E2E Test Strategy:
  - Mock ONLY the LLM boundary (llm_invoke, unfinished_prompt, postprocess)
  - Exercise the rest of the pipeline for real: CLI parsing, construct_paths,
    prompt reading, file writing, and (once added) function arg validation
  - Verify that the pipeline detects the mismatch

The test should:
  - FAIL on the current buggy code (no validation exists)
  - PASS once the fix adds _validate_python_function_args() to the pipeline

Issue: https://github.com/gltanaka/pdd/issues/625
"""

import textwrap
from pathlib import Path
from unittest.mock import patch

import click
import pytest

from pdd import DEFAULT_STRENGTH, DEFAULT_TIME


# ---------------------------------------------------------------------------
# Fixtures: buggy and correct generated code
# ---------------------------------------------------------------------------

# The exact buggy code from issue #625: 3 args passed to 4-param function
BUGGY_GENERATED_CODE = textwrap.dedent('''\
    """Hackathon compliance module."""
    import requests
    from typing import Dict

    GITHUB_API_URL = "https://api.github.com"

    def on_submission_created(repo_url: str, github_token: str) -> dict:
        """Handle new submission."""
        headers = {"Authorization": f"token {github_token}"}
        # BUG: 3 args passed to function expecting 4
        readme_content = fetch_file_content(repo_url, 'README.md', github_token)
        return {"readme": readme_content}

    def fetch_file_content(owner: str, repo: str, path: str, headers: Dict[str, str]) -> str:
        """Fetch file content from GitHub."""
        url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/contents/{path}"
        response = requests.get(url, headers=headers)
        return response.text
''')

# Correct version: all calls match definitions
CORRECT_GENERATED_CODE = textwrap.dedent('''\
    """Hackathon compliance module."""
    import requests
    from typing import Dict

    GITHUB_API_URL = "https://api.github.com"

    def on_submission_created(repo_url: str, github_token: str) -> dict:
        """Handle new submission."""
        headers = {"Authorization": f"token {github_token}"}
        owner, repo = repo_url.split("/")[-2:]
        readme_content = fetch_file_content(owner, repo, 'README.md', headers)
        return {"readme": readme_content}

    def fetch_file_content(owner: str, repo: str, path: str, headers: Dict[str, str]) -> str:
        """Fetch file content from GitHub."""
        url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/contents/{path}"
        response = requests.get(url, headers=headers)
        return response.text
''')


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

def _mock_llm_invoke_factory(code_to_return: str):
    """Create a mock llm_invoke that returns the specified code."""
    def mock_llm_invoke(*args, **kwargs):
        """Mock LLM call returning predetermined code."""
        return {
            "result": code_to_return,
            "cost": 0.01,
            "model_name": "mock-model",
        }
    return mock_llm_invoke


def _mock_unfinished_prompt(*args, **kwargs):
    """Mock unfinished_prompt: always says generation is complete."""
    return ("Generation looks complete", True, 0.0, "mock-model")


def _mock_postprocess(llm_output, language, **kwargs):
    """Mock postprocess: pass through the code without LLM extraction."""
    return (llm_output, 0.0, "mock-model")


# ---------------------------------------------------------------------------
# E2E Test 1: code_generator_main direct invocation
# ---------------------------------------------------------------------------

class TestIssue625E2ECodeGeneratorMain:
    """
    E2E tests that call code_generator_main() directly with a Click context,
    exercising the full pipeline: prompt reading -> preprocessing -> LLM (mocked)
    -> postprocess (mocked) -> validation -> file writing.

    This follows the same pattern as test_real_generate_command in
    test_commands_generate.py, but with deterministic LLM output.
    """

    def test_e2e_buggy_code_should_be_detected(self, tmp_path, monkeypatch):
        """
        E2E Test: When the LLM generates code with mismatched function args
        (3 args to a 4-param function), the code_generator_main pipeline
        should detect and report the issue.

        Full code path exercised:
          prompt file -> code_generator_main -> construct_paths -> preprocess
          -> code_generator -> llm_invoke (mocked) -> unfinished_prompt (mocked)
          -> postprocess (mocked) -> [validation gap] -> file write

        Current buggy behavior:
          - The buggy code passes through all checks silently
          - fetch_file_content(repo_url, 'README.md', github_token) is written to disk
          - No error or warning about the 3-arg call to a 4-param function

        Expected fixed behavior:
          - _validate_python_function_args() detects the mismatch
          - Pipeline reports the issue (error, warning, or non-empty return)
        """
        import pdd
        pdd_package_dir = Path(pdd.__file__).parent
        monkeypatch.setenv("PDD_PATH", str(pdd_package_dir))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key-for-e2e-625")

        # Create a prompt file with a name that construct_paths can resolve
        prompt_file = tmp_path / "hackathon_compliance_python.prompt"
        prompt_file.write_text(
            "Generate a hackathon compliance checker that validates GitHub repositories.\n"
            "It should fetch file content from the GitHub API.\n"
        )

        output_file = tmp_path / "hackathon_compliance.py"

        # Create Click context (same pattern as test_real_generate_command)
        ctx = click.Context(click.Command("generate"))
        ctx.obj = {
            'force': True,
            'quiet': True,
            'verbose': False,
            'strength': DEFAULT_STRENGTH,
            'temperature': 0.0,
            'local': True,
            'output_cost': None,
            'review_examples': False,
            'time': DEFAULT_TIME,
            'context': None,
        }

        from pdd.code_generator_main import code_generator_main

        with patch('pdd.code_generator.llm_invoke', side_effect=_mock_llm_invoke_factory(BUGGY_GENERATED_CODE)), \
             patch('pdd.code_generator.unfinished_prompt', side_effect=_mock_unfinished_prompt), \
             patch('pdd.code_generator.postprocess', side_effect=_mock_postprocess):

            generated_code, was_incremental, total_cost, model_name = code_generator_main(
                ctx=ctx,
                prompt_file=str(prompt_file),
                output=str(output_file),
                original_prompt_file_path=None,
                force_incremental_flag=False,
            )

        # --------------- THE BUG CHECK ---------------
        # After the fix, the pipeline should detect that fetch_file_content
        # is called with 3 args but defined with 4 required parameters.
        #
        # The fix may manifest as:
        # 1. The output file not being written (validation prevented it)
        # 2. The buggy call being removed from the output
        # 3. An error signal in the return value (model_name == "error")
        # 4. The generated_code being empty (validation rejected it)

        file_content = output_file.read_text() if output_file.exists() else ""
        buggy_call_written = "fetch_file_content(repo_url, 'README.md', github_token)" in file_content

        # After the fix, at least one of these should be true:
        pipeline_detected_bug = (
            not output_file.exists()      # Validation prevented file write
            or not buggy_call_written      # Buggy call was removed/fixed
            or generated_code == ""        # Code generation was rejected
            or model_name == "error"       # Error was signaled in return value
        )

        assert pipeline_detected_bug, (
            f"BUG DETECTED (Issue #625): Generated Python code with mismatched "
            f"function calls passed through the 'pdd generate' pipeline without "
            f"any validation.\n\n"
            f"fetch_file_content(repo_url, 'README.md', github_token) calls a "
            f"4-parameter function with only 3 positional arguments, but the "
            f"pipeline wrote it to disk silently.\n\n"
            f"Output file exists: {output_file.exists()}\n"
            f"Buggy call in file: {buggy_call_written}\n"
            f"Generated code length: {len(generated_code)} chars\n"
            f"Model name: {model_name}\n\n"
            f"Expected: Pipeline should validate function argument consistency "
            f"after postprocessing (analogous to _validate_python_imports for "
            f"issue #572)."
        )

    def test_e2e_correct_code_passes_through(self, tmp_path, monkeypatch):
        """
        Regression guard: When all function calls match their definitions,
        the pipeline should write the code to disk without false positives.

        This test should PASS both before and after the fix.
        """
        import pdd
        pdd_package_dir = Path(pdd.__file__).parent
        monkeypatch.setenv("PDD_PATH", str(pdd_package_dir))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key-for-e2e-625")

        prompt_file = tmp_path / "hackathon_compliance_python.prompt"
        prompt_file.write_text(
            "Generate a hackathon compliance checker that validates GitHub repositories.\n"
        )

        output_file = tmp_path / "hackathon_compliance.py"

        ctx = click.Context(click.Command("generate"))
        ctx.obj = {
            'force': True,
            'quiet': True,
            'verbose': False,
            'strength': DEFAULT_STRENGTH,
            'temperature': 0.0,
            'local': True,
            'output_cost': None,
            'review_examples': False,
            'time': DEFAULT_TIME,
            'context': None,
        }

        from pdd.code_generator_main import code_generator_main

        with patch('pdd.code_generator.llm_invoke', side_effect=_mock_llm_invoke_factory(CORRECT_GENERATED_CODE)), \
             patch('pdd.code_generator.unfinished_prompt', side_effect=_mock_unfinished_prompt), \
             patch('pdd.code_generator.postprocess', side_effect=_mock_postprocess):

            generated_code, was_incremental, total_cost, model_name = code_generator_main(
                ctx=ctx,
                prompt_file=str(prompt_file),
                output=str(output_file),
                original_prompt_file_path=None,
                force_incremental_flag=False,
            )

        # Correct code should be written to disk successfully
        assert output_file.exists(), (
            f"Output file should be written for code with correct function calls.\n"
            f"Generated code length: {len(generated_code)} chars"
        )

        file_content = output_file.read_text()
        assert "fetch_file_content(owner, repo, 'README.md', headers)" in file_content, (
            "Correct function call should be present in output file"
        )


# ---------------------------------------------------------------------------
# E2E Test 2: Full CLI path via CliRunner
# ---------------------------------------------------------------------------

class TestIssue625E2ECLIRunner:
    """
    E2E tests that exercise the full CLI path via Click's CliRunner.

    This is the most end-to-end approach: it exercises the actual command
    line interface that users interact with, including CLI argument parsing,
    the generate command dispatcher, and the full code_generator_main pipeline.
    """

    def test_e2e_cli_generate_detects_wrong_function_args(self, tmp_path, monkeypatch):
        """
        E2E Test: `pdd --local --force generate <prompt> --output <output>`
        should detect mismatched function arguments in generated Python code.

        User scenario:
          1. User has a prompt file for generating a hackathon compliance module
          2. Runs: pdd generate hackathon_compliance.prompt --output hackathon.py
          3. LLM generates code with fetch_file_content called with wrong arg count
          4. Expected: Pipeline catches the mismatch before writing to disk
          5. Actual (bug): Pipeline writes buggy code silently
        """
        import pdd
        from pdd import cli

        pdd_package_dir = Path(pdd.__file__).parent
        monkeypatch.setenv("PDD_PATH", str(pdd_package_dir))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key-for-e2e-625")
        monkeypatch.setenv("PDD_FORCE_LOCAL", "1")

        prompt_file = tmp_path / "hackathon_compliance_python.prompt"
        prompt_file.write_text(
            "Generate a hackathon compliance checker that validates GitHub repos.\n"
        )

        output_file = tmp_path / "hackathon_compliance.py"

        from click.testing import CliRunner

        with patch('pdd.core.cli.auto_update'), \
             patch('pdd.code_generator.llm_invoke', side_effect=_mock_llm_invoke_factory(BUGGY_GENERATED_CODE)), \
             patch('pdd.code_generator.unfinished_prompt', side_effect=_mock_unfinished_prompt), \
             patch('pdd.code_generator.postprocess', side_effect=_mock_postprocess):

            runner = CliRunner()
            result = runner.invoke(cli.cli, [
                "--local", "--force", "--quiet",
                "generate",
                str(prompt_file),
                "--output", str(output_file),
                "--exclude-tests",
            ])

        combined_output = result.output or ""

        # Check if the pipeline detected the function argument mismatch
        file_content = output_file.read_text() if output_file.exists() else ""
        buggy_call_written = "fetch_file_content(repo_url, 'README.md', github_token)" in file_content

        # After the fix, the pipeline should signal the mismatch via at least one of:
        # - Non-zero exit code
        # - Validation message in output
        # - Refusing to write the buggy file
        pipeline_detected_bug = (
            result.exit_code != 0
            or not output_file.exists()
            or not buggy_call_written
            or any(
                phrase in combined_output.lower()
                for phrase in ["function", "argument", "mismatch", "validate"]
            )
        )

        assert pipeline_detected_bug, (
            f"BUG DETECTED (Issue #625): `pdd generate` wrote Python code with "
            f"mismatched function calls to disk without any detection.\n\n"
            f"The generated code calls fetch_file_content with 3 args, but the "
            f"function requires 4 parameters. The pipeline should catch this.\n\n"
            f"Exit code: {result.exit_code}\n"
            f"Output file exists: {output_file.exists()}\n"
            f"Buggy call in file: {buggy_call_written}\n"
            f"CLI output ({len(combined_output)} chars): {combined_output[:500]}\n\n"
            f"Expected: Post-generation validation (analogous to "
            f"_validate_python_imports for issue #572) should detect the mismatch."
        )

    def test_e2e_cli_generate_correct_code_succeeds(self, tmp_path, monkeypatch):
        """
        Regression guard: `pdd generate` with correct function calls should
        write the output file successfully.
        """
        import pdd
        from pdd import cli

        pdd_package_dir = Path(pdd.__file__).parent
        monkeypatch.setenv("PDD_PATH", str(pdd_package_dir))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key-for-e2e-625")
        monkeypatch.setenv("PDD_FORCE_LOCAL", "1")

        prompt_file = tmp_path / "hackathon_compliance_python.prompt"
        prompt_file.write_text(
            "Generate a hackathon compliance checker that validates GitHub repos.\n"
        )

        output_file = tmp_path / "hackathon_compliance.py"

        from click.testing import CliRunner

        with patch('pdd.core.cli.auto_update'), \
             patch('pdd.code_generator.llm_invoke', side_effect=_mock_llm_invoke_factory(CORRECT_GENERATED_CODE)), \
             patch('pdd.code_generator.unfinished_prompt', side_effect=_mock_unfinished_prompt), \
             patch('pdd.code_generator.postprocess', side_effect=_mock_postprocess):

            runner = CliRunner()
            result = runner.invoke(cli.cli, [
                "--local", "--force", "--quiet",
                "generate",
                str(prompt_file),
                "--output", str(output_file),
                "--exclude-tests",
            ])

        # Correct code should be written successfully
        assert output_file.exists(), (
            f"Output file should be created for correct code.\n"
            f"Exit code: {result.exit_code}\n"
            f"Output: {result.output[:500]}"
        )

        file_content = output_file.read_text()
        assert "fetch_file_content(owner, repo, 'README.md', headers)" in file_content, (
            "Correct function call should be in the output file"
        )
