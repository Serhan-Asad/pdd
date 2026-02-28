"""Tests for function argument count validation in generated code (issue #625).

When pdd generates a Python file, function calls within that file should be
consistent with function definitions in the same file. The validation
function _validate_python_function_args() detects positional argument count
mismatches such as:
- Too few positional arguments passed to a function
- Too many positional arguments passed to a function (when no *args)

Calls that use keyword arguments are skipped since positional count alone
cannot reliably validate them.
"""

import textwrap

import pytest


@pytest.fixture
def validate_fn():
    """Get the _validate_python_function_args function or fail with clear message."""
    try:
        from pdd.sync_orchestration import _validate_python_function_args
        return _validate_python_function_args
    except ImportError:
        pytest.fail(
            "Issue #625: _validate_python_function_args not found in "
            "pdd.sync_orchestration. Post-generation validation for function "
            "argument consistency has not been implemented yet."
        )


# --- Test: Exact bug from issue #625 ---


def test_issue625_exact_bug_too_few_args(tmp_path, validate_fn):
    """Exact bug from issue #625: fetch_file_content called with 3 args but needs 4."""
    code = textwrap.dedent('''\
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
    code_file = tmp_path / "hackathon_compliance.py"
    code_file.write_text(code)

    issues = validate_fn(code_file)
    assert len(issues) > 0, (
        "Validator should detect that fetch_file_content is called with 3 args "
        "but defined with 4 required params (issue #625)"
    )


# --- Tests: Too few arguments ---


def test_one_arg_missing(tmp_path, validate_fn):
    """Function defined with 2 required params, called with 1 arg."""
    code = textwrap.dedent('''\
        def process(data: str, mode: str) -> str:
            """Process data."""
            return f"{data}-{mode}"

        result = process("hello")
    ''')
    code_file = tmp_path / "test_code.py"
    code_file.write_text(code)

    issues = validate_fn(code_file)
    assert len(issues) > 0, "Should detect 1 arg passed to 2-param function"


def test_zero_args_to_required_param_function(tmp_path, validate_fn):
    """Function defined with 1 required param, called with 0 args."""
    code = textwrap.dedent('''\
        def transform(value: int) -> int:
            """Transform value."""
            return value * 2

        result = transform()
    ''')
    code_file = tmp_path / "test_code.py"
    code_file.write_text(code)

    issues = validate_fn(code_file)
    assert len(issues) > 0, "Should detect 0 args passed to 1-param function"


# --- Test: Too many arguments ---


def test_too_many_args(tmp_path, validate_fn):
    """Function defined with 2 params, called with 3 args."""
    code = textwrap.dedent('''\
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = add(1, 2, 3)
    ''')
    code_file = tmp_path / "test_code.py"
    code_file.write_text(code)

    issues = validate_fn(code_file)
    assert len(issues) > 0, "Should detect 3 args passed to 2-param function"


# --- Tests: Correct calls (no false positives) ---


def test_correct_arg_count(tmp_path, validate_fn):
    """All function calls match their definitions — no issues expected."""
    code = textwrap.dedent('''\
        def fetch_file_content(owner: str, repo: str, path: str, headers: dict) -> str:
            """Fetch file content."""
            return f"{owner}/{repo}/{path}"

        content = fetch_file_content("user", "repo", "README.md", {"auth": "token"})
    ''')
    code_file = tmp_path / "test_code.py"
    code_file.write_text(code)

    issues = validate_fn(code_file)
    assert len(issues) == 0, "Correct function calls should not be flagged"


def test_mixed_correct_and_wrong_calls(tmp_path, validate_fn):
    """Multiple call sites — some correct, some wrong. Should flag the wrong one."""
    code = textwrap.dedent('''\
        def fetch_file_content(owner: str, repo: str, path: str, headers: dict) -> str:
            """Fetch file content."""
            return f"{owner}/{repo}/{path}"

        # Correct calls
        content1 = fetch_file_content("user", "repo", "README.md", {"auth": "token"})
        content2 = fetch_file_content("org", "project", "setup.py", {"auth": "key"})

        # Wrong call (3 args instead of 4)
        content3 = fetch_file_content("https://github.com/user/repo", "README.md", "token")
    ''')
    code_file = tmp_path / "test_code.py"
    code_file.write_text(code)

    issues = validate_fn(code_file)
    assert len(issues) > 0, "Should detect the wrong call even among correct ones"


# --- Tests: Default parameters ---


def test_optional_params_not_flagged(tmp_path, validate_fn):
    """Calling without optional args (that have defaults) is fine."""
    code = textwrap.dedent('''\
        def connect(host: str, port: int = 8080, timeout: float = 30.0) -> None:
            """Connect to server."""
            pass

        connect("localhost")
        connect("localhost", 9090)
        connect("localhost", 9090, 60.0)
    ''')
    code_file = tmp_path / "test_code.py"
    code_file.write_text(code)

    issues = validate_fn(code_file)
    assert len(issues) == 0, "Calls using default params should not be flagged"


def test_too_few_even_with_defaults(tmp_path, validate_fn):
    """Function has 1 required + 2 optional params, called with 0 args."""
    code = textwrap.dedent('''\
        def connect(host: str, port: int = 8080, timeout: float = 30.0) -> None:
            """Connect to server."""
            pass

        connect()  # Missing required 'host'
    ''')
    code_file = tmp_path / "test_code.py"
    code_file.write_text(code)

    issues = validate_fn(code_file)
    assert len(issues) > 0, "Should detect 0 args when 1 required param exists"


# --- Test: *args and **kwargs ---


def test_star_args_accepts_any_count(tmp_path, validate_fn):
    """Functions with *args should accept any number of positional args."""
    code = textwrap.dedent('''\
        def log(*messages: str) -> None:
            """Log messages."""
            for msg in messages:
                print(msg)

        log()
        log("hello")
        log("hello", "world", "!")
    ''')
    code_file = tmp_path / "test_code.py"
    code_file.write_text(code)

    issues = validate_fn(code_file)
    assert len(issues) == 0, "*args functions should accept any number of args"


# --- Test: Method calls ---


def test_method_calls_not_validated(tmp_path, validate_fn):
    """Method calls (obj.method()) should not be validated against free functions."""
    code = textwrap.dedent('''\
        def process(data: str) -> str:
            """Process data."""
            return data.upper()

        class Handler:
            def process(self, data: str, mode: str) -> str:
                """Handle data."""
                return f"{data}-{mode}"

        result = process("hello")  # Correct: 1 arg to 1-param function
    ''')
    code_file = tmp_path / "test_code.py"
    code_file.write_text(code)

    issues = validate_fn(code_file)
    assert len(issues) == 0, "Should not confuse method definitions with function calls"


# --- Test: Edge cases ---


def test_syntax_error_handled_gracefully(tmp_path, validate_fn):
    """Files with syntax errors should return empty list (not crash)."""
    code = "def foo(:\n    pass\n"
    code_file = tmp_path / "bad_syntax.py"
    code_file.write_text(code)

    issues = validate_fn(code_file)
    assert issues == [], "Syntax errors should be handled gracefully"
