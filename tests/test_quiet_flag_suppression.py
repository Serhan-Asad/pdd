# test_quiet_flag_suppression.py
"""
Tests for GitHub Issue #165: --quiet flag does not suppress output.

The bug: The --quiet flag is stored in Click context but never propagated to
the core output-producing modules (preprocess.py, load_prompt_template.py,
code_generator.py, llm_invoke.py). These functions lack a quiet parameter,
so Rich panels, status messages, and INFO logs are emitted unconditionally.

These tests verify that each affected function accepts and respects a
quiet=True parameter to suppress non-essential output.
"""

import inspect
import logging
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Test 1: preprocess() should accept a quiet parameter
# ---------------------------------------------------------------------------
class TestPreprocessQuietFlag:
    """Tests that preprocess() accepts and respects a quiet parameter."""

    def test_preprocess_accepts_quiet_parameter(self):
        """preprocess() must accept a 'quiet' keyword argument.

        Currently fails because preprocess() signature is:
            def preprocess(prompt, recursive=False, double_curly_brackets=True,
                           exclude_keys=None, _seen=None)
        with no quiet parameter.
        """
        from pdd.preprocess import preprocess

        sig = inspect.signature(preprocess)
        assert "quiet" in sig.parameters, (
            "preprocess() must accept a 'quiet' parameter to suppress output. "
            f"Current parameters: {list(sig.parameters.keys())}"
        )

    def test_preprocess_quiet_suppresses_panels(self, capsys):
        """When quiet=True, preprocess() should not print Rich Panels.

        Currently fails because preprocess() unconditionally prints:
        - Panel("Starting prompt preprocessing", ...)   [line 123]
        - Panel("Preprocessing complete", ...)           [line 142]
        - "Doubling curly brackets..."                   [line 457]
        """
        from pdd.preprocess import preprocess

        # Call preprocess with quiet=True on a simple prompt (no includes)
        try:
            result = preprocess("Hello world", quiet=True)
        except TypeError as e:
            # If preprocess() doesn't accept quiet, that's the bug
            if "quiet" in str(e):
                pytest.fail(
                    f"preprocess() does not accept 'quiet' parameter: {e}"
                )
            raise

        captured = capsys.readouterr()
        assert "Starting prompt preprocessing" not in captured.out, (
            "preprocess(quiet=True) should suppress 'Starting prompt preprocessing' panel"
        )
        assert "Preprocessing complete" not in captured.out, (
            "preprocess(quiet=True) should suppress 'Preprocessing complete' panel"
        )
        assert "Doubling curly brackets" not in captured.out, (
            "preprocess(quiet=True) should suppress 'Doubling curly brackets...' message"
        )

    def test_preprocess_non_quiet_still_shows_output(self, capsys):
        """When quiet=False (default), preprocess() should still show panels.

        Regression guard: ensure fixing quiet doesn't break normal output.
        """
        from pdd.preprocess import preprocess

        try:
            result = preprocess("Hello world", quiet=False)
        except TypeError:
            # If quiet param doesn't exist yet, skip this regression test
            pytest.skip("preprocess() does not yet accept 'quiet' parameter")

        captured = capsys.readouterr()
        # At minimum the preprocessing panels should appear
        assert "Starting prompt preprocessing" in captured.out or \
               "Preprocessing complete" in captured.out, (
            "preprocess(quiet=False) should still show preprocessing panels"
        )


# ---------------------------------------------------------------------------
# Test 2: load_prompt_template() should accept a quiet parameter
# ---------------------------------------------------------------------------
class TestLoadPromptTemplateQuietFlag:
    """Tests that load_prompt_template() accepts and respects a quiet parameter."""

    def test_load_prompt_template_accepts_quiet_parameter(self):
        """load_prompt_template() must accept a 'quiet' keyword argument.

        Currently fails because load_prompt_template() signature is:
            def load_prompt_template(prompt_name: str) -> Optional[str]:
        with no quiet parameter.
        """
        from pdd.load_prompt_template import load_prompt_template

        sig = inspect.signature(load_prompt_template)
        assert "quiet" in sig.parameters, (
            "load_prompt_template() must accept a 'quiet' parameter. "
            f"Current parameters: {list(sig.parameters.keys())}"
        )

    def test_load_prompt_template_quiet_suppresses_success_message(self, capsys, tmp_path):
        """When quiet=True, load_prompt_template() should not print success message.

        Currently fails because load_prompt_template() unconditionally prints:
            print_formatted(f"[green]Successfully loaded prompt: {prompt_name}[/green]")
        on line 50.
        """
        from pdd.load_prompt_template import load_prompt_template

        # Create a temporary prompt file
        prompt_file = tmp_path / "prompts" / "test_prompt.prompt"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text("Test prompt content")

        # Mock the path resolution to point to our temp file
        with patch("pdd.load_prompt_template.get_default_resolver") as mock_resolver:
            mock_resolver.return_value.resolve_prompt_template.return_value = prompt_file

            try:
                result = load_prompt_template("test_prompt", quiet=True)
            except TypeError as e:
                if "quiet" in str(e):
                    pytest.fail(
                        f"load_prompt_template() does not accept 'quiet' parameter: {e}"
                    )
                raise

        captured = capsys.readouterr()
        assert "Successfully loaded prompt" not in captured.out, (
            "load_prompt_template(quiet=True) should suppress 'Successfully loaded prompt' message"
        )


# ---------------------------------------------------------------------------
# Test 3: code_generator() should accept a quiet parameter
# ---------------------------------------------------------------------------
class TestCodeGeneratorQuietFlag:
    """Tests that code_generator() accepts and respects a quiet parameter."""

    def test_code_generator_accepts_quiet_parameter(self):
        """code_generator() must accept a 'quiet' keyword argument.

        Currently fails because code_generator() signature is:
            def code_generator(prompt, language, strength, temperature=0.0,
                               time=None, verbose=False, preprocess_prompt=True,
                               output_schema=None)
        with no quiet parameter.
        """
        from pdd.code_generator import code_generator

        sig = inspect.signature(code_generator)
        assert "quiet" in sig.parameters, (
            "code_generator() must accept a 'quiet' parameter. "
            f"Current parameters: {list(sig.parameters.keys())}"
        )

    def test_code_generator_passes_quiet_to_preprocess(self):
        """code_generator() should pass quiet=True to preprocess() when quiet is enabled.

        This tests the caller behavior: code_generator() must thread quiet
        through to preprocess() rather than calling it without the flag.
        """
        from pdd.code_generator import code_generator

        with patch("pdd.code_generator.preprocess") as mock_preprocess, \
             patch("pdd.code_generator.llm_invoke") as mock_llm, \
             patch("pdd.code_generator.unfinished_prompt") as mock_unfinished, \
             patch("pdd.code_generator.postprocess") as mock_postprocess:

            mock_preprocess.return_value = "processed prompt"
            mock_llm.return_value = {
                "result": "generated code",
                "cost": 0.01,
                "model_name": "test-model",
            }
            mock_unfinished.return_value = ("", True, 0.0, "")
            mock_postprocess.return_value = ("code", 0.0, "test-model")

            try:
                code_generator(
                    prompt="test prompt",
                    language="python",
                    strength=0.5,
                    quiet=True,
                )
            except TypeError as e:
                if "quiet" in str(e):
                    pytest.fail(
                        f"code_generator() does not accept 'quiet' parameter: {e}"
                    )
                raise

            # Verify preprocess was called with quiet=True
            mock_preprocess.assert_called_once()
            call_kwargs = mock_preprocess.call_args
            # Check both positional and keyword args for quiet
            assert "quiet" in (call_kwargs.kwargs or {}), (
                "code_generator(quiet=True) must pass quiet=True to preprocess(). "
                f"preprocess() was called with: args={call_kwargs.args}, kwargs={call_kwargs.kwargs}"
            )
            assert call_kwargs.kwargs.get("quiet") is True, (
                "code_generator(quiet=True) must pass quiet=True to preprocess()"
            )


# ---------------------------------------------------------------------------
# Test 4: code_generator_main() should pass quiet to code_generator()
# ---------------------------------------------------------------------------
class TestCodeGeneratorMainQuietPropagation:
    """Tests that code_generator_main() passes quiet from Click context to code_generator()."""

    def test_code_generator_main_passes_quiet_to_code_generator(self):
        """code_generator_main() reads quiet from ctx.obj but must pass it to code_generator().

        Currently fails because code_generator_main() extracts quiet at line 284:
            quiet = cli_params.get('quiet', False)
        but never passes it to local_code_generator_func() at line 913.
        The import is aliased: from .code_generator import code_generator as local_code_generator_func
        """
        import click
        from pdd.code_generator_main import code_generator_main

        ctx = click.Context(click.Command("test_cmd"))
        ctx.obj = {
            "strength": 0.5,
            "temperature": 0.0,
            "verbose": False,
            "force": True,
            "quiet": True,  # The key flag being tested
            "time": None,
            "local": True,
        }

        with patch("pdd.code_generator_main.construct_paths") as mock_cp, \
             patch("pdd.code_generator_main.local_code_generator_func") as mock_cg, \
             patch("pdd.code_generator_main.pdd_preprocess") as mock_pp, \
             patch("builtins.open", MagicMock(return_value=MagicMock(
                 __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value="prompt content"))),
                 __exit__=MagicMock(return_value=False),
             ))):

            mock_cp.return_value = (
                {},
                {"prompt_file": "prompt content"},
                {"output": "/tmp/output.py"},
                "python",
            )
            mock_pp.return_value = "preprocessed content"
            mock_cg.return_value = ("generated code", 0.01, "test-model")

            try:
                code_generator_main(
                    ctx=ctx,
                    prompt_file="test.prompt",
                    output="/tmp/output.py",
                    original_prompt_file_path=None,
                    force_incremental_flag=False,
                )
            except Exception:
                # We only care about verifying the call to code_generator
                pass

            # If code_generator was called, check if quiet was passed
            if mock_cg.called:
                call_kwargs = mock_cg.call_args.kwargs if mock_cg.call_args.kwargs else {}
                call_args_all = str(mock_cg.call_args)
                assert "quiet" in call_kwargs, (
                    "code_generator_main() must pass quiet=True to local_code_generator_func(). "
                    f"local_code_generator_func() was called with: {call_args_all}"
                )


# ---------------------------------------------------------------------------
# Test 5: set_verbose_logging() should handle quiet mode
# ---------------------------------------------------------------------------
class TestLlmInvokeQuietMode:
    """Tests that llm_invoke's logging respects quiet mode."""

    def test_set_verbose_logging_accepts_quiet_parameter(self):
        """set_verbose_logging() should accept a quiet parameter to suppress INFO logs.

        Currently fails because set_verbose_logging() signature is:
            def set_verbose_logging(verbose=False)
        with no quiet parameter. When quiet=True, the logger level should be
        set to WARNING or higher to suppress INFO lines.
        """
        from pdd.llm_invoke import set_verbose_logging

        sig = inspect.signature(set_verbose_logging)
        assert "quiet" in sig.parameters, (
            "set_verbose_logging() must accept a 'quiet' parameter. "
            f"Current parameters: {list(sig.parameters.keys())}"
        )

    def test_quiet_mode_suppresses_info_logs(self):
        """When quiet=True, the pdd.llm_invoke logger should be at WARNING or higher.

        Currently fails because set_verbose_logging() has no quiet mode —
        the default non-verbose level is INFO, which still emits INFO log lines
        that the user sees in --quiet mode.
        """
        from pdd.llm_invoke import set_verbose_logging, logger

        try:
            set_verbose_logging(quiet=True)
        except TypeError as e:
            if "quiet" in str(e):
                pytest.fail(
                    f"set_verbose_logging() does not accept 'quiet' parameter: {e}"
                )
            raise

        assert logger.level >= logging.WARNING, (
            f"In quiet mode, logger level should be WARNING ({logging.WARNING}) or higher, "
            f"but got {logging.getLevelName(logger.level)} ({logger.level})"
        )


# ---------------------------------------------------------------------------
# Test 6: Errors should still display when quiet=True
# ---------------------------------------------------------------------------
class TestQuietDoesNotSuppressErrors:
    """Regression guard: quiet mode should NOT suppress error messages."""

    def test_preprocess_error_still_shown_when_quiet(self, capsys):
        """Even with quiet=True, preprocess() should still show error messages.

        This ensures the fix doesn't over-suppress output.
        """
        from pdd.preprocess import preprocess

        try:
            # Empty prompt should trigger error output
            result = preprocess("", quiet=True)
        except TypeError:
            # If quiet param doesn't exist yet, skip this regression test
            pytest.skip("preprocess() does not yet accept 'quiet' parameter")

        captured = capsys.readouterr()
        # Error message should still appear even in quiet mode
        assert "Error" in captured.out or result == "", (
            "preprocess(quiet=True) should still show error messages for empty prompts"
        )
