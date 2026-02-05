"""
E2E Test for Issue #71: Gemini CLI requires -p flag for non-interactive mode

This E2E test exercises the full code path from agentic workflow to verify that
the Gemini CLI command is constructed correctly with the -p flag for non-interactive mode.

The Bug:
--------
Gemini CLI v0.27.0 changed its behavior - positional arguments now default to interactive mode,
causing `pdd bug` and other agentic commands to hang when calling the gemini CLI.

Root Cause:
-----------
In `pdd/agentic_common.py:490-495`, the `_run_with_provider` function passes the prompt as a
positional argument without the `-p` flag:

```python
cmd = [
    cli_path,
    f"Read the file {prompt_path.name} for your full instructions and execute them.",
    "--yolo",
    "--output-format", "json"
]
```

The Gemini CLI now shows this message:
> ℹ Positional arguments now default to interactive mode. To run in non-interactive mode,
> use the --prompt (-p) flag.

Expected Fix:
-------------
Add the `-p` flag to force non-interactive mode:

```python
cmd = [
    cli_path,
    "-p", f"Read the file {prompt_path.name} for your full instructions and execute them.",
    "--yolo",
    "--output-format", "json"
]
```

E2E Test Strategy:
------------------
This test:
1. Mocks subprocess.run at the E2E boundary (where commands are executed)
2. Calls run_agentic_task() which internally uses _run_with_provider()
3. Captures the actual command that would be passed to subprocess.run
4. Verifies that the `-p` flag is present in the command
5. Does NOT mock the buggy component (_run_with_provider) - we test the real code path

The test should:
- FAIL on the current buggy code (no `-p` flag in command)
- PASS once the bug is fixed (`.p` flag present in command)

Issue: https://github.com/Serhan-Asad/pdd/issues/71
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def set_pdd_path(monkeypatch):
    """Set PDD_PATH to the pdd package directory for all tests in this module.

    This is required because construct_paths uses PDD_PATH to find configuration files.
    """
    import pdd
    pdd_package_dir = Path(pdd.__file__).parent
    monkeypatch.setenv("PDD_PATH", str(pdd_package_dir))


@pytest.mark.e2e
class TestGeminiPFlagE2E:
    """
    E2E tests for Issue #71: Verify Gemini CLI commands use -p flag for non-interactive mode.

    These tests exercise the real code path through run_agentic_task() and _run_with_provider()
    to verify the command construction is correct.
    """

    def test_agentic_task_uses_p_flag_for_gemini(self, tmp_path, monkeypatch):
        """
        E2E Test: run_agentic_task() with Google provider should construct Gemini CLI
        commands with the -p flag for non-interactive mode.

        This is the primary E2E test for Issue #71. It exercises the full code path
        from run_agentic_task() through _run_with_provider() and captures the actual
        command that would be executed.

        Expected behavior (after fix):
        - Command includes: [cli_path, "-p", "Read the file ...", "--yolo", "--output-format", "json"]
        - The -p flag forces non-interactive mode (required for Gemini CLI v0.27.0+)
        - The instruction text follows the -p flag (not a raw file path)

        Bug behavior (Issue #71):
        - Command does NOT include -p flag
        - Positional argument defaults to interactive mode
        - Gemini CLI hangs waiting for interactive input
        - All agentic commands fail when using Google provider
        """
        # 1. Create a working directory with a prompt file
        cwd = tmp_path / "work"
        cwd.mkdir()

        # 2. Mock the Gemini CLI binary location
        mock_cli_path = "/usr/local/bin/gemini"

        # 3. Set up environment to use Google provider ONLY
        # Provide API key so Google provider is detected as available
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key-for-e2e")

        # 4. Track the actual subprocess command that would be executed
        captured_commands = []
        captured_kwargs = []

        def mock_subprocess_run(cmd, **kwargs):
            """Mock subprocess.run to capture the command without executing it."""
            captured_commands.append(cmd)
            captured_kwargs.append(kwargs)

            # Return a successful result with valid, non-empty JSON output
            # The output must be non-empty to avoid false positive detection
            mock_result = MagicMock()
            mock_result.returncode = 0
            # Return actual content to avoid false positive detection
            mock_result.stdout = '{"output": "Successfully completed the task with meaningful output", "cost": 0.01}'
            mock_result.stderr = ""
            return mock_result

        # 5. Mock _find_cli_binary to only return gemini as available
        def mock_find_cli_binary(name):
            """Only gemini CLI should be available."""
            if name == "gemini":
                return mock_cli_path
            return None

        # 6. Patch subprocess.run and _find_cli_binary at the point where commands are executed
        with patch("pdd.agentic_common.subprocess.run", side_effect=mock_subprocess_run):
            with patch("pdd.agentic_common._find_cli_binary", side_effect=mock_find_cli_binary):
                # Import after patching to ensure we get the mocked version
                from pdd.agentic_common import run_agentic_task

                # 7. Call run_agentic_task with an instruction
                # This exercises the full code path including _run_with_provider()
                instruction = "Test instruction for Gemini CLI to execute"
                success, output, cost, provider = run_agentic_task(
                    instruction=instruction,
                    cwd=cwd,
                    timeout=60
                )

        # 8. Verify subprocess.run was called
        assert len(captured_commands) > 0, (
            "Expected subprocess.run to be called at least once, but it was not called.\n"
            "This indicates a setup issue with the test."
        )

        # 9. Get the command that was passed to subprocess.run
        actual_cmd = captured_commands[0]

        # 10. THE KEY ASSERTIONS - Check for -p flag

        # The command MUST include the -p flag for non-interactive mode
        assert "-p" in actual_cmd, (
            f"BUG DETECTED (Issue #71): Gemini CLI command does NOT include -p flag!\n\n"
            f"Gemini CLI v0.27.0+ requires the -p flag for non-interactive mode.\n"
            f"Without -p, positional arguments default to interactive mode, causing\n"
            f"agentic commands to hang waiting for user input.\n\n"
            f"Expected command format:\n"
            f"  [cli_path, '-p', 'Read the file ...', '--yolo', '--output-format', 'json']\n\n"
            f"Actual command:\n"
            f"  {actual_cmd}\n\n"
            f"Impact: All agentic commands (pdd bug, pdd change, etc.) fail with Google provider."
        )

        # 11. Verify -p flag positioning and value
        p_flag_index = actual_cmd.index("-p")

        # The -p flag should be followed by the instruction text
        assert p_flag_index + 1 < len(actual_cmd), (
            f"BUG: -p flag found but no instruction text follows it.\n"
            f"Command: {actual_cmd}"
        )

        instruction_text = actual_cmd[p_flag_index + 1]

        # The instruction should contain "Read the file" directive
        assert "Read the file" in instruction_text, (
            f"BUG: -p flag is present but instruction text is malformed.\n"
            f"Expected: 'Read the file .agentic_prompt_XXXXX.txt for your full instructions...'\n"
            f"Actual: '{instruction_text}'\n"
            f"Full command: {actual_cmd}"
        )

        # Verify the instruction refers to a prompt file (.agentic_prompt_*.txt)
        assert ".agentic_prompt_" in instruction_text and ".txt" in instruction_text, (
            f"BUG: Instruction text does not reference the agentic prompt file.\n"
            f"Expected pattern: '.agentic_prompt_XXXXX.txt'\n"
            f"Instruction text: '{instruction_text}'\n"
            f"Full command: {actual_cmd}"
        )

        # 12. Verify other required flags are present
        assert "--yolo" in actual_cmd, (
            f"Missing --yolo flag in Gemini command.\n"
            f"Command: {actual_cmd}"
        )

        assert "--output-format" in actual_cmd, (
            f"Missing --output-format flag in Gemini command.\n"
            f"Command: {actual_cmd}"
        )

        assert "json" in actual_cmd, (
            f"Missing 'json' value for --output-format in Gemini command.\n"
            f"Command: {actual_cmd}"
        )

    def test_anthropic_provider_not_affected(self, tmp_path, monkeypatch):
        """
        E2E Test: Verify that Anthropic provider is not affected by the Gemini CLI change.

        This negative test ensures that the fix for Issue #71 is specific to the Google
        provider and doesn't break other providers.

        The Anthropic provider uses stdin for prompt content and a different CLI structure,
        so it should not be affected by Gemini CLI v0.27.0's positional argument change.
        """
        # 1. Create a working directory
        cwd = tmp_path / "work"
        cwd.mkdir()

        # 2. Mock the Claude CLI binary location
        mock_cli_path = "/usr/local/bin/claude"

        # 3. Mock _find_cli_binary to only return claude as available
        def mock_find_cli_binary(name):
            """Only claude CLI should be available."""
            if name == "claude":
                return mock_cli_path
            return None

        # 4. Track the actual subprocess command that would be executed
        captured_commands = []

        def mock_subprocess_run(cmd, **kwargs):
            """Mock subprocess.run to capture the command."""
            captured_commands.append(cmd)

            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '{"output": "Successfully completed the task with meaningful output", "cost": 0.01}'
            mock_result.stderr = ""
            return mock_result

        # 5. Patch subprocess.run and _find_cli_binary
        with patch("pdd.agentic_common.subprocess.run", side_effect=mock_subprocess_run):
            with patch("pdd.agentic_common._find_cli_binary", side_effect=mock_find_cli_binary):
                from pdd.agentic_common import run_agentic_task

                # 6. Call run_agentic_task with Anthropic provider
                instruction = "Test instruction for Anthropic Claude CLI"
                success, output, cost, provider = run_agentic_task(
                    instruction=instruction,
                    cwd=cwd,
                    timeout=60
                )

        # 6. Verify subprocess.run was called
        assert len(captured_commands) > 0, "Expected subprocess.run to be called"

        # 7. Get the command that was passed to subprocess.run
        actual_cmd = captured_commands[0]

        # 8. Verify Anthropic command structure is different
        # Anthropic uses "-p", "-" (stdin) not "-p", "instruction text"
        if "-p" in actual_cmd:
            p_flag_index = actual_cmd.index("-p")
            # Anthropic should have "-" after -p (for stdin)
            assert p_flag_index + 1 < len(actual_cmd)
            stdin_marker = actual_cmd[p_flag_index + 1]
            assert stdin_marker == "-", (
                f"Anthropic provider should use '-p -' for stdin input.\n"
                f"Command: {actual_cmd}"
            )

    def test_openai_provider_not_affected(self, tmp_path, monkeypatch):
        """
        E2E Test: Verify that OpenAI provider is not affected by the Gemini CLI change.

        This negative test ensures that the fix for Issue #71 is specific to the Google
        provider and doesn't break the OpenAI provider.

        The OpenAI provider uses "exec --full-auto" command structure, so it should
        not be affected by Gemini CLI v0.27.0's positional argument change.
        """
        # 1. Create a working directory
        cwd = tmp_path / "work"
        cwd.mkdir()

        # 2. Mock the Codex CLI binary location (OpenAI uses codex CLI)
        mock_cli_path = "/usr/local/bin/codex"

        # 3. Set up environment to use OpenAI provider ONLY
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-e2e")

        # 4. Track the actual subprocess command
        captured_commands = []

        def mock_subprocess_run(cmd, **kwargs):
            """Mock subprocess.run to capture the command."""
            captured_commands.append(cmd)

            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = '{"output": "Successfully completed the task with meaningful output", "cost": 0.01}'
            mock_result.stderr = ""
            return mock_result

        # 5. Mock _find_cli_binary to only return codex as available
        def mock_find_cli_binary(name):
            """Only codex CLI should be available (OpenAI provider uses codex CLI)."""
            if name == "codex":
                return mock_cli_path
            return None

        # 6. Patch subprocess.run and _find_cli_binary
        with patch("pdd.agentic_common.subprocess.run", side_effect=mock_subprocess_run):
            with patch("pdd.agentic_common._find_cli_binary", side_effect=mock_find_cli_binary):
                from pdd.agentic_common import run_agentic_task

                # 7. Call run_agentic_task with OpenAI provider
                instruction = "Test instruction for OpenAI CLI"
                success, output, cost, provider = run_agentic_task(
                    instruction=instruction,
                    cwd=cwd,
                    timeout=60
                )

        # 8. Verify subprocess.run was called
        assert len(captured_commands) > 0, "Expected subprocess.run to be called"

        # 9. Get the command that was passed to subprocess.run
        actual_cmd = captured_commands[0]

        # 10. Verify OpenAI command structure uses "exec" subcommand
        assert "exec" in actual_cmd, (
            f"OpenAI provider should use 'exec' subcommand.\n"
            f"Command: {actual_cmd}"
        )

        assert "--full-auto" in actual_cmd, (
            f"OpenAI provider should use '--full-auto' flag.\n"
            f"Command: {actual_cmd}"
        )
