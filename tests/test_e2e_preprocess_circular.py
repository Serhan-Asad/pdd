"""
E2E test for issue #526: Circular <include> tags without --recursive cause infinite loop.

This test exercises the full CLI path (`pdd preprocess`) rather than calling
the preprocess() function directly, verifying the bug from the user's perspective.
"""

import os
import threading

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


def test_e2e_circular_include_cli_does_not_hang(circular_prompt_dir):
    """Issue #526 E2E: `pdd preprocess` with circular includes must not hang.

    Reproduces the exact user scenario from the issue:
      pdd --force preprocess prompts/a_python.prompt
    with two files that include each other (A→B→A).

    On buggy code, the CLI enters an infinite loop and never returns.
    This test uses a thread + timeout to detect the hang.
    """
    from pdd.cli import cli

    tmp_path, file_a, _ = circular_prompt_dir

    runner = CliRunner(mix_stderr=False)
    result_holder = [None]

    def run_cli():
        result_holder[0] = runner.invoke(
            cli,
            ["--force", "preprocess", str(file_a)],
            env={**os.environ, "PDD_PATH": str(tmp_path)},
            catch_exceptions=True,
        )

    t = threading.Thread(target=run_cli, daemon=True)
    t.start()
    t.join(timeout=10)  # 10s generous timeout; normal preprocess takes <1s

    if t.is_alive():
        pytest.fail(
            "Issue #526 E2E: `pdd --force preprocess` entered an infinite loop "
            "with circular includes (A→B→A). The CLI never terminated. "
            "Expected: error message and non-zero exit code."
        )

    result = result_holder[0]
    assert result is not None, "CLI invocation returned no result"

    # Once fixed, should exit with non-zero and mention circular/cycle
    # For now, we just verify it doesn't hang (the fail above catches the bug)


def test_e2e_self_referencing_include_cli_does_not_hang(tmp_path):
    """Issue #526 E2E: `pdd preprocess` with self-referencing include must not hang."""
    from pdd.cli import cli

    prompt_file = tmp_path / "self_ref.prompt"
    prompt_file.write_text(f"I reference myself <include>{prompt_file}</include>")

    runner = CliRunner(mix_stderr=False)
    result_holder = [None]

    def run_cli():
        result_holder[0] = runner.invoke(
            cli,
            ["--force", "preprocess", str(prompt_file)],
            env={**os.environ, "PDD_PATH": str(tmp_path)},
            catch_exceptions=True,
        )

    t = threading.Thread(target=run_cli, daemon=True)
    t.start()
    t.join(timeout=10)

    if t.is_alive():
        pytest.fail(
            "Issue #526 E2E: `pdd --force preprocess` entered an infinite loop "
            "with a self-referencing include. The CLI never terminated."
        )

    result = result_holder[0]
    assert result is not None, "CLI invocation returned no result"
