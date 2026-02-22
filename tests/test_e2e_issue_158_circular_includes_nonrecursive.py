"""
E2E Test for Issue #158: Circular include cycle detection missing in non-recursive preprocess path (v2).

This test exercises the full CLI path: `pdd preprocess` (default, no --recursive flag) with
real circular prompt files on disk. No mocking of the preprocessor. This verifies the
user-facing behavior: running `pdd preprocess` on circular includes must produce an error
and exit promptly, not hang forever in an infinite loop.

Issue #158 extends the scenarios from the #157 E2E tests by covering:
- Backtick-style circular includes (```<file>```) — not just XML <include> tags
- Three-file backtick cycles (A→B→C→A)
- Mixed circular + non-circular includes in the same prompt
- Self-referencing backtick includes
- Diamond pattern regression guard (non-circular, must still work)

Root cause: In non-recursive mode (the default), `_seen` is only populated inside
the `if recursive:` branches (lines 272-274 and 179-181 of pdd/preprocess.py).
The `while prev_text != current_text` convergence loop then runs forever on circular
includes because the cycle check at lines 226/174 never fires.
"""

import os
import subprocess
import sys
import pytest
from pathlib import Path


# Invoke pdd-cli via Python -c to avoid conflict with the system 'pdd' date utility.
PDD_CMD = [sys.executable, "-c", "from pdd.cli import cli; cli()"]

# Root of the pdd-cli source tree (editable install).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

# Timeout in seconds — generous for CI, but a non-hanging preprocess should finish in <2s.
# The bug causes the process to hang forever, so any timeout proves the bug.
E2E_TIMEOUT = 10


def _run_pdd_preprocess(
    prompt_file: str,
    output_file: str,
    recursive: bool = False,
    timeout: int = E2E_TIMEOUT,
) -> subprocess.CompletedProcess:
    """Run `pdd preprocess` via subprocess with a timeout.

    Returns CompletedProcess on success, raises TimeoutExpired if the process hangs.
    """
    cmd = PDD_CMD + ["--force", "preprocess", prompt_file]
    if recursive:
        cmd.append("--recursive")
    cmd.extend(["--output", output_file])

    env = {**os.environ, "PDD_FORCE": "1", "PYTHONPATH": _PROJECT_ROOT}

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


class TestIssue158CircularIncludesNonRecursiveE2E:
    """E2E: `pdd preprocess` (no --recursive) on circular includes must error, not hang.

    These tests cover backtick includes, mixed scenarios, and additional patterns
    not covered by the #157 E2E tests (which focus on XML <include> tags).
    """

    # -- Backtick circular includes ------------------------------------------

    def test_circular_backtick_ab_cli_must_not_hang(self, tmp_path, monkeypatch):
        """A→B→A circular backtick includes via CLI must error, not hang the terminal.

        The backtick include syntax (```<file>```) uses process_backtick_includes(),
        which has the same _seen bug as process_include_tags().
        """
        monkeypatch.chdir(tmp_path)

        a_file = tmp_path / "a.txt"
        b_file = tmp_path / "b.txt"
        prompt_file = tmp_path / "circ_backtick_python.prompt"
        output_file = tmp_path / "output.txt"

        a_file.write_text("AlphaContent\n```<b.txt>```\n")
        b_file.write_text("BetaContent\n```<a.txt>```\n")
        prompt_file.write_text("```<a.txt>```\n")

        try:
            result = _run_pdd_preprocess(str(prompt_file), str(output_file))
        except subprocess.TimeoutExpired:
            pytest.fail(
                f"pdd preprocess hung for {E2E_TIMEOUT}s on circular A→B→A backtick "
                f"includes (non-recursive mode). Bug #158: _seen is never populated "
                f"when recursive=False in process_backtick_includes()."
            )

        # If it didn't hang, it should have exited with an error
        assert result.returncode != 0, (
            f"pdd preprocess exited with code 0 on circular backtick includes "
            f"(expected non-zero). stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )

    def test_self_referencing_backtick_include_cli_must_not_hang(self, tmp_path, monkeypatch):
        """A file that includes itself via backtick syntax must error via CLI, not hang."""
        monkeypatch.chdir(tmp_path)

        self_file = tmp_path / "selfref_python.prompt"
        output_file = tmp_path / "output.txt"

        self_file.write_text("SelfContent\n```<selfref_python.prompt>```\n")

        try:
            result = _run_pdd_preprocess(str(self_file), str(output_file))
        except subprocess.TimeoutExpired:
            pytest.fail(
                f"pdd preprocess hung for {E2E_TIMEOUT}s on self-referencing backtick "
                f"include (non-recursive mode). Bug #158: no cycle detection when "
                f"recursive=False."
            )

        assert result.returncode != 0, (
            f"pdd preprocess exited with code 0 on self-referencing backtick include "
            f"(expected non-zero). stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )

    def test_three_file_cycle_backtick_cli_must_not_hang(self, tmp_path, monkeypatch):
        """A→B→C→A three-file cycle via backtick includes must error, not hang."""
        monkeypatch.chdir(tmp_path)

        (tmp_path / "r.txt").write_text("Red\n```<g.txt>```\n")
        (tmp_path / "g.txt").write_text("Green\n```<b.txt>```\n")
        (tmp_path / "b.txt").write_text("Blue\n```<r.txt>```\n")
        prompt_file = tmp_path / "cycle3_backtick_python.prompt"
        output_file = tmp_path / "output.txt"
        prompt_file.write_text("```<r.txt>```\n")

        try:
            result = _run_pdd_preprocess(str(prompt_file), str(output_file))
        except subprocess.TimeoutExpired:
            pytest.fail(
                f"pdd preprocess hung for {E2E_TIMEOUT}s on A→B→C→A backtick cycle "
                f"(non-recursive mode). Bug #158."
            )

        assert result.returncode != 0, (
            f"pdd preprocess exited with code 0 on 3-file backtick cycle "
            f"(expected non-zero). stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )

    # -- Mixed scenarios -----------------------------------------------------

    def test_mixed_circular_and_noncircular_cli_must_not_hang(self, tmp_path, monkeypatch):
        """Prompt with both circular and non-circular includes must detect the cycle.

        The non-circular include (safe.txt) should not prevent detection of the
        circular pair (cyc1.txt ↔ cyc2.txt).
        """
        monkeypatch.chdir(tmp_path)

        (tmp_path / "safe.txt").write_text("SafeContent\n")
        (tmp_path / "cyc1.txt").write_text("<include>cyc2.txt</include>\n")
        (tmp_path / "cyc2.txt").write_text("<include>cyc1.txt</include>\n")
        prompt_file = tmp_path / "mixed_python.prompt"
        output_file = tmp_path / "output.txt"
        prompt_file.write_text(
            "<include>safe.txt</include>\n<include>cyc1.txt</include>\n"
        )

        try:
            result = _run_pdd_preprocess(str(prompt_file), str(output_file))
        except subprocess.TimeoutExpired:
            pytest.fail(
                f"pdd preprocess hung for {E2E_TIMEOUT}s on mixed circular + "
                f"non-circular includes (non-recursive mode). Bug #158: the "
                f"non-circular include doesn't prevent the cycle from hanging."
            )

        assert result.returncode != 0, (
            f"pdd preprocess exited with code 0 on mixed circular + non-circular "
            f"includes (expected non-zero). stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )

    # -- Regression guards (non-circular, must PASS) -------------------------

    def test_diamond_with_content_cli_nonrecursive_still_works(self, tmp_path, monkeypatch):
        """Diamond pattern (A→B, A→C, B→D, C→D) is NOT circular and must still work.

        D is legitimately included twice (via B and via C). This verifies that
        cycle detection, once fixed, does not produce false positives on diamond
        include graphs.
        """
        monkeypatch.chdir(tmp_path)

        (tmp_path / "b.txt").write_text("BContent\n<include>d.txt</include>\n")
        (tmp_path / "c.txt").write_text("CContent\n<include>d.txt</include>\n")
        (tmp_path / "d.txt").write_text("SharedLeaf\n")
        prompt_file = tmp_path / "diamond_python.prompt"
        output_file = tmp_path / "output.txt"
        prompt_file.write_text(
            "<include>b.txt</include>\n<include>c.txt</include>\n"
        )

        try:
            result = _run_pdd_preprocess(str(prompt_file), str(output_file))
        except subprocess.TimeoutExpired:
            pytest.fail("pdd preprocess hung on non-circular diamond includes.")

        # Should succeed — no cycles
        assert result.returncode == 0, (
            f"pdd preprocess failed on non-circular diamond includes (expected success). "
            f"stderr: {result.stderr[:500]}"
        )

        # Verify the output contains the included content
        output_content = output_file.read_text()
        assert "BContent" in output_content, "Included file 'b.txt' content missing"
        assert "CContent" in output_content, "Included file 'c.txt' content missing"
        assert output_content.count("SharedLeaf") == 2, (
            "D is included twice via B and C — that's fine, not circular"
        )
