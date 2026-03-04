"""
E2E Test for Issue #197: _parse_front_matter silently swallows YAML with CRLF line endings.

When prompt content contains \\r\\n line endings (e.g., from git autocrlf, network
responses, or files read with newline=''), _parse_front_matter returns (None, text),
causing YAML front matter to be treated as prompt content and sent to the LLM.

NOTE: Python's open('r') normalizes \\r\\n → \\n by default, so the standard file-read
path in code_generator_main (line 299) doesn't trigger this bug. However, the function
_parse_front_matter should be robust against CRLF regardless of how content arrives.
This E2E test opens the file with newline='' to preserve CRLF, simulating scenarios
where content retains its original line endings (git output, API responses, etc.).

The test should FAIL on buggy code and PASS once the fix is applied.
"""

import builtins
import os
import pathlib
import subprocess

import click
import pytest
import requests
from unittest.mock import MagicMock

from pdd.code_generator_main import code_generator_main


# --- Constants ---
MOCK_CODE = "def add(a, b):\n    return a + b"
MOCK_COST = 0.001
MOCK_MODEL = "mock-model"


@pytest.fixture
def mock_ctx():
    """Click context with local mode enabled."""
    ctx = MagicMock(spec=click.Context)
    ctx.obj = {
        "local": True,
        "strength": 0.5,
        "temperature": 0.0,
        "time": 0.25,
        "verbose": False,
        "force": True,
        "quiet": False,
    }
    return ctx


@pytest.fixture(autouse=True)
def mock_external_deps(monkeypatch):
    """Mock external dependencies that every code_generator_main call needs."""
    monkeypatch.setattr(
        "pdd.code_generator_main.CloudConfig.get_jwt_token",
        MagicMock(return_value="fake"),
    )
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.json.return_value = {
        "generatedCode": MOCK_CODE,
        "totalCost": MOCK_COST,
        "modelName": MOCK_MODEL,
    }
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    monkeypatch.setattr(
        "pdd.code_generator_main.requests.post",
        MagicMock(return_value=mock_resp),
    )
    monkeypatch.setattr(
        "pdd.code_generator_main.subprocess.run",
        MagicMock(
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
        ),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        MagicMock(
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
        ),
    )
    monkeypatch.setattr("pdd.code_generator_main.console.print", MagicMock())


@pytest.fixture
def prompt_env(tmp_path, monkeypatch):
    """Temp directory with PDD_PATH set."""
    pdd_root = tmp_path / "pdd_root" / "data"
    pdd_root.mkdir(parents=True)
    (pdd_root / "llm_model.csv").write_text("model,cost\nmock,0.01")
    monkeypatch.setenv("PDD_PATH", str(tmp_path / "pdd_root"))
    monkeypatch.setenv("PDD_FORCE_LOCAL", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    return tmp_path


def _make_crlf_aware_open(target_path: str):
    """Return a patched open() that uses newline='' for the target file.

    This preserves \\r\\n in the file content, simulating scenarios where
    prompt content retains Windows line endings (git autocrlf, API responses,
    binary-mode reads decoded to str, etc.).
    """
    _real_open = builtins.open

    def patched_open(file, mode="r", *args, **kwargs):
        # For the target prompt file in text-read mode, preserve CRLF
        try:
            if (
                mode == "r"
                and "newline" not in kwargs
                and str(pathlib.Path(file).resolve()) == target_path
            ):
                kwargs["newline"] = ""
        except Exception:
            pass
        return _real_open(file, mode, *args, **kwargs)

    return patched_open


class TestCRLFFrontMatterE2E:
    """E2E: CRLF prompt files should have their front matter parsed, not leaked to the LLM."""

    def test_crlf_front_matter_language_override_and_prompt_stripping(
        self, mock_ctx, prompt_env, monkeypatch
    ):
        """
        Write a CRLF prompt with 'language: ruby' in front matter, patch open()
        to preserve CRLF, run code_generator_main, and verify:
        1. The language passed to the local generator is 'ruby' (from front matter)
        2. The prompt sent to the LLM does NOT contain raw YAML metadata

        BUG: _parse_front_matter uses startswith('---\\n') which fails on '---\\r\\n',
        returning (None, text). This causes:
        - language stays as 'python' (from construct_paths) instead of 'ruby'
        - Raw YAML front matter leaks into the prompt sent to the LLM
        """
        # 1. Write a CRLF prompt file
        prompt_file = prompt_env / "prompts" / "crlf_test_python.prompt"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_bytes(
            b"---\r\nname: crlf_module\r\nlanguage: ruby\r\n---\r\n"
            b"Generate a function that adds two numbers.\r\n"
        )

        output_path = str(prompt_env / "output" / "crlf_output.rb")
        prompt_path_resolved = str(prompt_file.resolve())

        # 2. Patch open() to preserve CRLF for this file
        monkeypatch.setattr(
            builtins, "open", _make_crlf_aware_open(prompt_path_resolved)
        )

        # 3. Mock construct_paths
        def mock_construct_paths(**kwargs):
            content = prompt_file.read_text(encoding="utf-8")
            return (
                {},
                {"prompt_file": content},
                {"output": output_path},
                "python",  # construct_paths detects python from filename
            )

        monkeypatch.setattr(
            "pdd.code_generator_main.construct_paths", mock_construct_paths
        )

        # 4. Capture what gets sent to the local code generator
        captured_calls = []

        def mock_local_generator(**kwargs):
            captured_calls.append(kwargs)
            return (MOCK_CODE, MOCK_COST, MOCK_MODEL)

        monkeypatch.setattr(
            "pdd.code_generator_main.local_code_generator_func",
            mock_local_generator,
        )

        monkeypatch.setattr(
            "pdd.code_generator_main.pdd_preprocess",
            lambda text, **kw: text,
        )

        # 5. Run the full code_generator_main path
        code, incremental, cost, model = code_generator_main(
            mock_ctx, str(prompt_file), output_path, None, False
        )

        assert code == MOCK_CODE, f"Generation should succeed, got: {code}"
        assert len(captured_calls) == 1, "Local generator should be called exactly once"

        call_kwargs = captured_calls[0]

        # 6. BUG CHECK #1: language should be 'ruby' (from front matter override)
        assert call_kwargs["language"] == "ruby", (
            f"BUG DETECTED (Issue #197): Front matter 'language: ruby' was ignored!\n"
            f"CRLF line endings caused _parse_front_matter to return None,\n"
            f"so the language override from front matter was lost.\n"
            f"Expected language='ruby', got language='{call_kwargs['language']}'"
        )

        # 7. BUG CHECK #2: prompt should NOT contain front matter YAML
        prompt_sent = call_kwargs["prompt"]
        assert "name: crlf_module" not in prompt_sent, (
            f"BUG DETECTED (Issue #197): YAML metadata leaked into LLM prompt!\n"
            f"Expected: front matter parsed and stripped\n"
            f"Got prompt containing: 'name: crlf_module'\n"
            f"Full prompt: {prompt_sent[:300]}"
        )

    def test_lf_front_matter_still_works(
        self, mock_ctx, prompt_env, monkeypatch
    ):
        """
        Regression guard: LF-only front matter must continue to work
        even when open() preserves line endings.
        """
        prompt_file = prompt_env / "prompts" / "lf_test_python.prompt"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(
            "---\nname: lf_module\nlanguage: ruby\n---\n"
            "Generate a function that adds two numbers.\n"
        )

        output_path = str(prompt_env / "output" / "lf_output.rb")
        prompt_path_resolved = str(prompt_file.resolve())

        # Patch open() to preserve line endings (LF stays as LF)
        monkeypatch.setattr(
            builtins, "open", _make_crlf_aware_open(prompt_path_resolved)
        )

        def mock_construct_paths(**kwargs):
            content = prompt_file.read_text(encoding="utf-8")
            return (
                {},
                {"prompt_file": content},
                {"output": output_path},
                "python",
            )

        monkeypatch.setattr(
            "pdd.code_generator_main.construct_paths", mock_construct_paths
        )

        captured_calls = []

        def mock_local_generator(**kwargs):
            captured_calls.append(kwargs)
            return (MOCK_CODE, MOCK_COST, MOCK_MODEL)

        monkeypatch.setattr(
            "pdd.code_generator_main.local_code_generator_func",
            mock_local_generator,
        )

        monkeypatch.setattr(
            "pdd.code_generator_main.pdd_preprocess",
            lambda text, **kw: text,
        )

        code, incremental, cost, model = code_generator_main(
            mock_ctx, str(prompt_file), output_path, None, False
        )

        assert code == MOCK_CODE
        assert len(captured_calls) == 1
        assert captured_calls[0]["language"] == "ruby", (
            f"LF front matter language override failed: got '{captured_calls[0]['language']}'"
        )
        assert "name: lf_module" not in captured_calls[0]["prompt"]
