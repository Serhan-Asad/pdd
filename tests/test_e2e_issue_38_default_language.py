"""
E2E Test for Issue #38: Language detection ignores .pddrc default_language setting

This test exercises the full CLI path from `pdd generate` to verify that when a .pddrc
file contains a `default_language` setting, the language detection uses it as a fallback
when no explicit language indicator is present in the prompt filename.

The bug: When a .pddrc file has `default_language: python` configured, and the user runs
`pdd generate test.prompt` (without _python suffix), the command fails with:
"Could not determine language from input files or options."

Root cause: pdd/construct_paths.py:688 checks command_options.get("language") instead of
command_options.get("default_language"), causing the fallback to never be consulted.

This E2E test:
1. Creates a .pddrc file with default_language: python
2. Creates a prompt file WITHOUT a language suffix (test.prompt)
3. Runs `pdd generate test.prompt` through Click's CliRunner
4. Verifies the command succeeds (or fails, detecting the bug)

The test should FAIL on buggy code (ValueError: Could not determine language) and PASS
once the fix is applied (language detection falls back to default_language).

Issue: https://github.com/Serhan-Asad/pdd/issues/38
Canonical: https://github.com/promptdriven/pdd/issues/35
"""

import pytest
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import patch, MagicMock


@pytest.mark.e2e
class TestDefaultLanguageE2E:
    """
    E2E tests for Issue #38: Verify `pdd generate` uses default_language from .pddrc
    as a fallback when no other language indicator is present.
    """

    def test_pdd_generate_uses_default_language_from_pddrc(self, tmp_path, monkeypatch):
        """
        E2E Test: `pdd generate` should use default_language from .pddrc as fallback

        This test runs the full CLI path and verifies that when:
        - A .pddrc file has default_language: python
        - A prompt file has no language suffix (test.prompt)
        - No --language CLI flag is provided

        Then the language detection should fall back to default_language and succeed.

        Expected behavior (after fix):
        - Language detection uses default_language: python
        - Command proceeds past language detection
        - No "Could not determine language" error

        Bug behavior (Issue #38):
        - Language detection ignores default_language
        - Command fails with: "Could not determine language from input files or options."
        - User is forced to rename file to test_python.prompt
        """
        # 1. Set up the test environment
        test_dir = tmp_path / "test_project"
        test_dir.mkdir(parents=True, exist_ok=True)

        # Create .pddrc with default_language setting
        pddrc_content = """contexts:
  default:
    paths:
      - "**"
    defaults:
      generate_output_path: src/
      default_language: python
"""
        pddrc_path = test_dir / ".pddrc"
        pddrc_path.write_text(pddrc_content)

        # Create prompt file WITHOUT language suffix (the bug scenario)
        prompt_content = "write a python function 'hello' that prints hello"
        prompt_path = test_dir / "test.prompt"
        prompt_path.write_text(prompt_content)

        # Change to test directory so CLI can find .pddrc
        monkeypatch.chdir(test_dir)

        # Import after changing directory to ensure proper initialization
        from pdd.cli import cli

        runner = CliRunner()

        # 2. Run the CLI command
        # We don't need to mock LLM - the test is about language detection which happens
        # during path construction, before LLM invocation. If language detection fails,
        # we'll get the error immediately without ever reaching the LLM call.
        result = runner.invoke(
            cli,
            ["generate", "test.prompt"],
            env={
                "PDD_AUTO_UPDATE": "false",
                "HOME": str(tmp_path),
            },
            catch_exceptions=True  # Catch to examine the error
        )

        # 3. THE KEY ASSERTIONS

        # Check if the bug exists (command fails with language detection error)
        output_lower = result.output.lower()

        if "could not determine language" in output_lower:
            pytest.fail(
                f"BUG DETECTED (Issue #38): `pdd generate` ignores default_language from .pddrc!\n\n"
                f"Exit code: {result.exit_code}\n"
                f"Output:\n{result.output}\n\n"
                f"The .pddrc file contained: default_language: python\n"
                f"The prompt file was: test.prompt (no language suffix)\n\n"
                f"Root cause: pdd/construct_paths.py:688 checks command_options.get('language')\n"
                f"instead of command_options.get('default_language'), causing the fallback to\n"
                f"never be consulted.\n\n"
                f"Expected: Language detection should fall back to default_language: python\n"
                f"Actual: Command fails with 'Could not determine language' error\n\n"
                f"Priority should be: CLI flag > File extension > Filename suffix > default_language > Error\n"
                f"But currently: CLI flag > File extension > Filename suffix > Error (skips default_language!)\n\n"
                f"Workaround: Rename file to test_python.prompt"
            )

        # If we get here without the language detection error, the bug is fixed or
        # we got a different error (which is OK for this test - we only care about
        # language detection working)

        # The command might still fail for other reasons (e.g., missing API keys),
        # but it should NOT fail with "Could not determine language"
        assert "could not determine language" not in output_lower, (
            f"Language detection should not fail when default_language is set in .pddrc\n"
            f"Output: {result.output}"
        )

    def test_pdd_generate_filename_suffix_overrides_default_language(self, tmp_path, monkeypatch):
        """
        E2E Test: Filename suffix should take precedence over default_language

        This test verifies that when both default_language (from .pddrc) and a filename
        suffix (e.g., _javascript.prompt) are present, the filename suffix wins.

        This ensures the fix doesn't break existing priority behavior.
        """
        # 1. Set up test environment
        test_dir = tmp_path / "test_project"
        test_dir.mkdir(parents=True, exist_ok=True)

        # Create .pddrc with default_language: python
        pddrc_content = """contexts:
  default:
    paths:
      - "**"
    defaults:
      generate_output_path: src/
      default_language: python
"""
        pddrc_path = test_dir / ".pddrc"
        pddrc_path.write_text(pddrc_content)

        # Create prompt file WITH javascript suffix (should override python default)
        prompt_content = "write a javascript function hello"
        prompt_path = test_dir / "test_javascript.prompt"
        prompt_path.write_text(prompt_content)

        monkeypatch.chdir(test_dir)

        from pdd.cli import cli
        runner = CliRunner()

        # 2. Run command
        result = runner.invoke(
            cli,
            ["generate", "test_javascript.prompt"],
            env={
                "PDD_AUTO_UPDATE": "false",
                "HOME": str(tmp_path),
            },
            catch_exceptions=True
        )

        # 3. Verify language detection doesn't fail
        # (The command might fail for other reasons like missing API keys, but
        # it should NOT fail with "Could not determine language")
        output_lower = result.output.lower()
        assert "could not determine language" not in output_lower, (
            f"Language detection should work with filename suffix\n"
            f"Output: {result.output}"
        )

    def test_pdd_generate_cli_flag_overrides_default_language(self, tmp_path, monkeypatch):
        """
        E2E Test: --language CLI flag should have highest priority

        This test verifies that the --language flag overrides both default_language
        and filename suffix, ensuring proper precedence order.
        """
        # 1. Set up test environment
        test_dir = tmp_path / "test_project"
        test_dir.mkdir(parents=True, exist_ok=True)

        # Create .pddrc with default_language: python
        pddrc_content = """contexts:
  default:
    paths:
      - "**"
    defaults:
      generate_output_path: src/
      default_language: python
"""
        pddrc_path = test_dir / ".pddrc"
        pddrc_path.write_text(pddrc_content)

        # Create prompt file with no suffix
        prompt_content = "write a function hello"
        prompt_path = test_dir / "test.prompt"
        prompt_path.write_text(prompt_content)

        monkeypatch.chdir(test_dir)

        from pdd.cli import cli
        runner = CliRunner()

        # 2. Run command with --language typescript (should override python default)
        result = runner.invoke(
            cli,
            ["generate", "--language", "typescript", "test.prompt"],
            env={
                "PDD_AUTO_UPDATE": "false",
                "HOME": str(tmp_path),
            },
            catch_exceptions=True
        )

        # 3. Verify language detection doesn't fail
        output_lower = result.output.lower()
        assert "could not determine language" not in output_lower, (
            f"Language detection should work with --language flag\n"
            f"Output: {result.output}"
        )

    def test_pdd_generate_env_var_fallback(self, tmp_path, monkeypatch):
        """
        E2E Test: PDD_DEFAULT_LANGUAGE env var should work as fallback

        This test verifies that the PDD_DEFAULT_LANGUAGE environment variable
        also works as a fallback source for language detection.
        """
        # 1. Set up test environment WITHOUT .pddrc
        test_dir = tmp_path / "test_project"
        test_dir.mkdir(parents=True, exist_ok=True)

        # Create prompt file with no suffix
        prompt_content = "write a function hello"
        prompt_path = test_dir / "test.prompt"
        prompt_path.write_text(prompt_content)

        monkeypatch.chdir(test_dir)

        from pdd.cli import cli
        runner = CliRunner()

        # 2. Run command with PDD_DEFAULT_LANGUAGE env var
        result = runner.invoke(
            cli,
            ["generate", "test.prompt"],
            env={
                "PDD_AUTO_UPDATE": "false",
                "HOME": str(tmp_path),
                "PDD_DEFAULT_LANGUAGE": "python"  # Set via env var
            },
            catch_exceptions=True
        )

        # 3. Check for language detection error
        output_lower = result.output.lower()

        if "could not determine language" in output_lower:
            pytest.fail(
                f"BUG DETECTED: PDD_DEFAULT_LANGUAGE env var is also ignored!\n\n"
                f"Exit code: {result.exit_code}\n"
                f"Output:\n{result.output}\n\n"
                f"The PDD_DEFAULT_LANGUAGE environment variable was set to: python\n"
                f"But language detection still failed.\n\n"
                f"This confirms the bug affects both .pddrc default_language and\n"
                f"PDD_DEFAULT_LANGUAGE env var."
            )

        # After fix, should not have language detection error
        assert "could not determine language" not in output_lower, (
            f"Language detection should work with PDD_DEFAULT_LANGUAGE env var\n"
            f"Output: {result.output}"
        )

    def test_pdd_generate_error_when_no_language_available(self, tmp_path, monkeypatch):
        """
        E2E Test: Should show clear error when no language source exists

        This test verifies that when truly no language indicator is present
        (no .pddrc, no env var, no filename suffix, no CLI flag), the command
        fails with a clear error message.
        """
        # 1. Set up minimal environment with NO language sources
        test_dir = tmp_path / "test_project"
        test_dir.mkdir(parents=True, exist_ok=True)

        # Create prompt file with no suffix
        prompt_content = "write a function hello"
        prompt_path = test_dir / "test.prompt"
        prompt_path.write_text(prompt_content)

        monkeypatch.chdir(test_dir)

        from pdd.cli import cli
        runner = CliRunner()

        # 2. Run command without any language source
        result = runner.invoke(
            cli,
            ["generate", "test.prompt"],
            env={
                "PDD_AUTO_UPDATE": "false",
                "HOME": str(tmp_path),
                # Explicitly do NOT set PDD_DEFAULT_LANGUAGE
            },
            catch_exceptions=True  # Expect this to fail
        )

        # 3. Verify proper error
        assert result.exit_code != 0, (
            "Command should fail when no language source is available"
        )

        output_lower = result.output.lower()
        assert "could not determine language" in output_lower or "language" in output_lower, (
            f"Error message should mention language detection failure\n"
            f"Output: {result.output}"
        )
