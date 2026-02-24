"""
Tests for GitHub Issue #166: --quiet flag does not suppress output.

The --quiet flag is defined in cli.py but never threaded through to the
functions that actually produce output:
  - preprocess() in pdd/preprocess.py prints Rich panels unconditionally
  - double_curly() in pdd/preprocess.py prints "Doubling curly brackets..." unconditionally
  - load_prompt_template() in pdd/load_prompt_template.py prints success messages unconditionally

These tests verify that quiet mode suppresses non-essential output in each
of these functions. All tests should FAIL on the current buggy code because
the functions do not accept a `quiet` parameter.
"""

import os
import io
import pytest
from unittest.mock import patch, mock_open, MagicMock


# ---------------------------------------------------------------------------
# Test 1: preprocess() suppresses Rich panels in quiet mode
# ---------------------------------------------------------------------------
def test_preprocess_suppresses_panels_in_quiet_mode(capsys) -> None:
    """preprocess(quiet=True) must not print 'Starting prompt preprocessing'
    or 'Preprocessing complete' Rich panels.

    Currently fails because preprocess() has no quiet parameter and
    unconditionally calls console.print(Panel(...)).
    """
    from pdd.preprocess import preprocess

    # A minimal prompt that triggers no includes — just the panel output
    with patch("pdd.preprocess.console") as mock_console:
        result = preprocess("Hello world", recursive=False,
                            double_curly_brackets=False, quiet=True)

    # The function should still return preprocessed text
    assert result == "Hello world"

    # In quiet mode, no Panel should have been printed
    for call in mock_console.print.call_args_list:
        args = call[0]
        for arg in args:
            assert not hasattr(arg, "renderable"), (
                f"Rich Panel printed in quiet mode: {arg}"
            )


# ---------------------------------------------------------------------------
# Test 2: preprocess() still prints panels when NOT in quiet mode (regression)
# ---------------------------------------------------------------------------
def test_preprocess_prints_panels_when_not_quiet() -> None:
    """When quiet is False (the default), preprocess() should still print
    its Rich panels — verifies we don't accidentally suppress all output.
    """
    from pdd.preprocess import preprocess

    with patch("pdd.preprocess.console") as mock_console:
        preprocess("Hello world", recursive=False,
                   double_curly_brackets=False)

    # At least one Panel should have been printed
    panel_printed = False
    for call in mock_console.print.call_args_list:
        args = call[0]
        for arg in args:
            if hasattr(arg, "renderable"):
                panel_printed = True
                break
    assert panel_printed, "Expected Rich Panel output when quiet mode is off"


# ---------------------------------------------------------------------------
# Test 3: load_prompt_template() suppresses success message in quiet mode
# ---------------------------------------------------------------------------
def test_load_prompt_template_suppresses_output_in_quiet_mode(
    monkeypatch, capsys
) -> None:
    """load_prompt_template(quiet=True) must not print 'Successfully loaded
    prompt' to stdout.

    Currently fails because load_prompt_template() has no quiet parameter.
    """
    from pdd.load_prompt_template import load_prompt_template

    monkeypatch.setenv("PDD_PATH", "/fake/project/path")
    prompt_content = "template content here"

    from pathlib import Path
    with patch.object(Path, "exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=prompt_content)):
            result = load_prompt_template("example_prompt", quiet=True)

    assert result == prompt_content

    captured = capsys.readouterr()
    assert "Successfully loaded prompt" not in captured.out, (
        "Success message printed despite quiet=True"
    )


# ---------------------------------------------------------------------------
# Test 4: preprocess() quiet mode preserves functional correctness
# ---------------------------------------------------------------------------
def test_preprocess_quiet_mode_preserves_output(capsys) -> None:
    """preprocess() with quiet=True must return the same preprocessed text
    as without quiet (no console noise, same functional result).

    Currently fails because preprocess() does not accept quiet parameter.
    """
    from pdd.preprocess import preprocess

    prompt = "Hello {name} world"

    with patch("pdd.preprocess.console"):
        result_normal = preprocess(prompt, recursive=False,
                                   double_curly_brackets=True)

    with patch("pdd.preprocess.console"):
        result_quiet = preprocess(prompt, recursive=False,
                                  double_curly_brackets=True, quiet=True)

    assert result_normal == result_quiet, (
        "quiet mode should not alter the preprocessed output"
    )


# ---------------------------------------------------------------------------
# Test 5: double_curly() suppresses its message in quiet mode
# ---------------------------------------------------------------------------
def test_double_curly_suppresses_message_in_quiet_mode() -> None:
    """double_curly(quiet=True) must not print 'Doubling curly brackets...'.

    Currently fails because double_curly() has no quiet parameter and
    unconditionally calls console.print("Doubling curly brackets...").
    """
    from pdd.preprocess import double_curly

    with patch("pdd.preprocess.console") as mock_console:
        result = double_curly("Hello {name}", quiet=True)

    # Verify no 'Doubling curly brackets...' message was printed
    for call in mock_console.print.call_args_list:
        args_str = " ".join(str(a) for a in call[0])
        assert "Doubling curly brackets" not in args_str, (
            "double_curly() printed message despite quiet=True"
        )
