"""
E2E test for issue #526: Circular <include> tags cause ValueError with cycle detection.

This test exercises the full CLI path (`pdd preprocess`) rather than calling
the preprocess() function directly, verifying the bug from the user's perspective.
"""

import os

import pytest
from click.testing import CliRunner


@pytest.fixture
def circular_prompt_dir(tmp_path):
    """Create a minimal directory with circular include prompt files."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()

    file_a = prompts / "a_python.prompt"
    file_b = prompts / "b_python.prompt"
    file_a.write_text(f"Create <include>{file_b}</include>")
    file_b.write_text(f"Create <include>{file_a}</include>")

    return tmp_path, file_a, file_b


def test_e2e_circular_include_cli_exits_with_error(circular_prompt_dir):
    """Issue #526 E2E: `pdd preprocess` with circular includes exits with error."""
    from pdd.cli import cli

    tmp_path, file_a, _ = circular_prompt_dir

    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(
        cli,
        ["--force", "preprocess", str(file_a)],
        env={**os.environ, "PDD_PATH": str(tmp_path)},
        catch_exceptions=True,
    )

    assert result.exit_code != 0 or "circular" in (result.output + result.stderr).lower(), (
        f"Expected non-zero exit or circular error message, got exit_code={result.exit_code}"
    )


def test_e2e_self_referencing_include_cli_exits_with_error(tmp_path):
    """Issue #526 E2E: `pdd preprocess` with self-referencing include exits with error."""
    from pdd.cli import cli

    prompt_file = tmp_path / "self_ref.prompt"
    prompt_file.write_text(f"I reference myself <include>{prompt_file}</include>")

    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(
        cli,
        ["--force", "preprocess", str(prompt_file)],
        env={**os.environ, "PDD_PATH": str(tmp_path)},
        catch_exceptions=True,
    )

    assert result.exit_code != 0 or "circular" in (result.output + result.stderr).lower(), (
        f"Expected non-zero exit or circular error message, got exit_code={result.exit_code}"
    )
