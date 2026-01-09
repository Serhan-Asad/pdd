import pytest
from rich.console import Console
from pdd import DEFAULT_STRENGTH
from pdd.generate_test import generate_test, _validate_inputs

# Test fixtures
@pytest.fixture
def valid_inputs():
    return {
        'prompt': 'Write a function to calculate factorial',
        'code': 'def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)',
        'strength': 0.5,
        'temperature': 0.5,
        'language': 'python'
    }

@pytest.fixture
def mock_console():
    return Console()

# Test successful generation
def test_generate_test_successful(valid_inputs):
    result = generate_test(**valid_inputs)
    assert isinstance(result, tuple)
    assert len(result) == 3
    unit_test, total_cost, model_name = result
    assert isinstance(unit_test, str)
    assert isinstance(total_cost, float)
    assert isinstance(model_name, str)
    assert total_cost >= 0
    assert len(model_name) > 0

# Test verbose output
def test_generate_test_verbose(valid_inputs):
    valid_inputs['verbose'] = True
    result = generate_test(**valid_inputs)
    assert isinstance(result, tuple)
    assert len(result) == 3

def test_validate_inputs_empty_prompt():
    with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
        _validate_inputs("", "code", DEFAULT_STRENGTH, 0.5, "python")


def test_validate_inputs_none_code():
    with pytest.raises(ValueError, match="Code must be a non-empty string"):
        _validate_inputs("prompt", None, DEFAULT_STRENGTH, 0.5, "python")


def test_validate_inputs_invalid_strength():
    with pytest.raises(ValueError, match="Strength must be a float between 0 and 1"):
        _validate_inputs("prompt", "code", 1.5, 0.5, "python")


def test_validate_inputs_invalid_temperature():
    with pytest.raises(ValueError, match="Temperature must be a float"):
        _validate_inputs("prompt", "code", DEFAULT_STRENGTH, "invalid", "python")


def test_validate_inputs_empty_language():
    with pytest.raises(ValueError, match="Language must be a non-empty string"):
        _validate_inputs("prompt", "code", DEFAULT_STRENGTH, 0.5, "")

# Test error handling
def test_generate_test_invalid_template(valid_inputs, monkeypatch):
    def mock_load_template(name):
        return None
    
    monkeypatch.setattr("pdd.generate_test.load_prompt_template", mock_load_template)
    
    with pytest.raises(ValueError, match="Failed to load generate_test_LLM prompt template"):
        generate_test(**valid_inputs)

# Test edge cases
def test_generate_test_minimum_values(valid_inputs):
    valid_inputs['strength'] = 0.31
    valid_inputs['temperature'] = 0.0
    result = generate_test(**valid_inputs)
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_generate_test_maximum_values(valid_inputs):
    valid_inputs['strength'] = 1.0
    valid_inputs['temperature'] = 1.0
    result = generate_test(**valid_inputs)
    assert isinstance(result, tuple)
    assert len(result) == 3

# Test different languages
def test_generate_test_different_languages(monkeypatch):
    # Avoid dependence on structured output in continuation by stubbing continue_generation
    def _stub_continue(formatted_input_prompt, llm_output, strength, temperature, time=0.25, language=None, verbose=False):
        return (llm_output, 0.0, "stub-model")
    monkeypatch.setattr("pdd.generate_test.continue_generation", _stub_continue)
    languages = ['python', 'javascript', 'java', 'cpp']
    for lang in languages:
        result = generate_test(
            prompt='Write a hello world function',
            code='print("Hello, World!")',
            strength=0.5,
            temperature=0.0,
            language=lang
        )
        assert isinstance(result, tuple)
        assert len(result) == 3


# -----------------------------------------------------------------------------
# Issue 212: TDD Mode Tests - Example-based Test Generation
# -----------------------------------------------------------------------------

def test_generate_test_tdd_mode_with_example_parameter(monkeypatch):
    """
    Test generate_test in TDD mode using 'example' parameter instead of 'code'.

    Issue 212: TDD workflow should generate tests from prompt + example.
    """
    # Stub out LLM calls to avoid API dependencies
    def _stub_llm_invoke(prompt, input_json, strength, temperature, time, verbose):
        # Verify we're in TDD mode by checking input_json
        assert input_json.get("example", "") != "", "TDD mode should have example content"
        assert input_json.get("code", "") == "", "TDD mode should have empty code"
        return {"result": "def test_example(): pass", "cost": 0.01, "model_name": "test_model"}

    def _stub_unfinished(prompt_text, strength, temperature, time, language, verbose):
        return ("complete", True, 0.0, "test_model")

    def _stub_postprocess(llm_output, language, strength, temperature, time, verbose):
        return (llm_output, 0.0, "test_model")

    monkeypatch.setattr("pdd.generate_test.llm_invoke", _stub_llm_invoke)
    monkeypatch.setattr("pdd.generate_test.unfinished_prompt", _stub_unfinished)
    monkeypatch.setattr("pdd.generate_test.postprocess", _stub_postprocess)

    result = generate_test(
        prompt="Create a calculator module",
        code=None,  # No code in TDD mode
        example="from calculator import add\nresult = add(5, 3)",  # Example provided
        strength=0.5,
        temperature=0.0,
        language="python"
    )

    assert isinstance(result, tuple)
    assert len(result) == 3
    unit_test, cost, model = result
    assert isinstance(unit_test, str)
    assert len(unit_test) > 0


def test_generate_test_mutual_exclusivity_both_code_and_example():
    """
    Test that providing both 'code' and 'example' raises ValueError.

    Issue 212: code and example parameters are mutually exclusive.
    """
    with pytest.raises(ValueError, match="mutually exclusive"):
        generate_test(
            prompt="Test prompt",
            code="def func(): pass",  # Both provided
            example="from module import func",  # Both provided
            strength=0.5,
            temperature=0.0,
            language="python"
        )


def test_generate_test_mutual_exclusivity_neither_code_nor_example():
    """
    Test that providing neither 'code' nor 'example' raises ValueError.

    Issue 212: Exactly one of code or example must be provided.
    """
    with pytest.raises(ValueError, match="Neither 'code' nor 'example' was provided"):
        generate_test(
            prompt="Test prompt",
            code=None,  # Neither provided
            example=None,  # Neither provided
            strength=0.5,
            temperature=0.0,
            language="python"
        )


def test_generate_test_tdd_mode_sends_different_input_json(monkeypatch):
    """
    Test that TDD mode sends different Input JSON to LLM than traditional mode.

    Issue 212: Verify TDD mode sends 'example' content, traditional sends 'code'.
    """
    captured_inputs = []

    def _capture_llm_invoke(prompt, input_json, strength, temperature, time, verbose):
        captured_inputs.append(input_json.copy())
        return {"result": "def test_func(): pass", "cost": 0.01, "model_name": "test_model"}

    def _stub_unfinished(prompt_text, strength, temperature, time, language, verbose):
        return ("complete", True, 0.0, "test_model")

    def _stub_postprocess(llm_output, language, strength, temperature, time, verbose):
        return (llm_output, 0.0, "test_model")

    monkeypatch.setattr("pdd.generate_test.llm_invoke", _capture_llm_invoke)
    monkeypatch.setattr("pdd.generate_test.unfinished_prompt", _stub_unfinished)
    monkeypatch.setattr("pdd.generate_test.postprocess", _stub_postprocess)

    # Test TDD mode
    generate_test(
        prompt="Create calculator",
        code=None,
        example="from calculator import add",
        strength=0.5,
        temperature=0.0,
        language="python"
    )

    tdd_input = captured_inputs[-1]
    assert tdd_input["code"] == "", "TDD mode should send empty code"
    assert tdd_input["example"] != "", "TDD mode should send example content"

    # Test traditional mode
    generate_test(
        prompt="Create calculator",
        code="def add(a, b): return a + b",
        example=None,
        strength=0.5,
        temperature=0.0,
        language="python"
    )

    traditional_input = captured_inputs[-1]
    assert traditional_input["code"] != "", "Traditional mode should send code content"
    assert traditional_input["example"] == "", "Traditional mode should send empty example"


def test_generate_test_tdd_mode_with_module_name_parameter(monkeypatch):
    """
    Test that TDD mode correctly passes module_name to LLM.

    Issue 212: module_name helps LLM generate correct import statements.
    """
    captured_inputs = []

    def _capture_llm_invoke(prompt, input_json, strength, temperature, time, verbose):
        captured_inputs.append(input_json.copy())
        return {"result": "def test_func(): pass", "cost": 0.01, "model_name": "test_model"}

    def _stub_unfinished(prompt_text, strength, temperature, time, language, verbose):
        return ("complete", True, 0.0, "test_model")

    def _stub_postprocess(llm_output, language, strength, temperature, time, verbose):
        return (llm_output, 0.0, "test_model")

    monkeypatch.setattr("pdd.generate_test.llm_invoke", _capture_llm_invoke)
    monkeypatch.setattr("pdd.generate_test.unfinished_prompt", _stub_unfinished)
    monkeypatch.setattr("pdd.generate_test.postprocess", _stub_postprocess)

    generate_test(
        prompt="Create calculator",
        code=None,
        example="from calculator import add",
        strength=0.5,
        temperature=0.0,
        language="python",
        module_name="calculator",  # Stripped from calculator_example
        source_file_path="calculator_example.py",
        test_file_path="test_calculator.py"
    )

    input_json = captured_inputs[-1]
    assert input_json["module_name"] == "calculator", \
        "module_name should be passed to LLM for correct imports"
    assert input_json["source_file_path"] == "calculator_example.py"
    assert input_json["test_file_path"] == "test_calculator.py"


def test_generate_test_tdd_mode_verbose_output_shows_mode(monkeypatch, capsys):
    """
    Test that verbose mode displays 'TDD (example-based)' mode indicator.

    Issue 212: User should see which mode is being used in verbose output.
    """
    def _stub_llm_invoke(prompt, input_json, strength, temperature, time, verbose):
        return {"result": "def test_func(): pass", "cost": 0.01, "model_name": "test_model"}

    def _stub_unfinished(prompt_text, strength, temperature, time, language, verbose):
        return ("complete", True, 0.0, "test_model")

    def _stub_postprocess(llm_output, language, strength, temperature, time, verbose):
        return (llm_output, 0.0, "test_model")

    monkeypatch.setattr("pdd.generate_test.llm_invoke", _stub_llm_invoke)
    monkeypatch.setattr("pdd.generate_test.unfinished_prompt", _stub_unfinished)
    monkeypatch.setattr("pdd.generate_test.postprocess", _stub_postprocess)

    generate_test(
        prompt="Create calculator",
        code=None,
        example="from calculator import add",
        strength=0.5,
        temperature=0.0,
        language="python",
        verbose=True  # Enable verbose output
    )

    captured = capsys.readouterr()
    assert "TDD (example-based)" in captured.out, \
        "Verbose output should indicate TDD mode"


def test_generate_test_traditional_mode_verbose_output_shows_mode(monkeypatch, capsys):
    """
    Test that verbose mode displays 'Traditional (code-based)' mode indicator.

    Issue 212: User should see which mode is being used in verbose output.
    """
    def _stub_llm_invoke(prompt, input_json, strength, temperature, time, verbose):
        return {"result": "def test_func(): pass", "cost": 0.01, "model_name": "test_model"}

    def _stub_unfinished(prompt_text, strength, temperature, time, language, verbose):
        return ("complete", True, 0.0, "test_model")

    def _stub_postprocess(llm_output, language, strength, temperature, time, verbose):
        return (llm_output, 0.0, "test_model")

    monkeypatch.setattr("pdd.generate_test.llm_invoke", _stub_llm_invoke)
    monkeypatch.setattr("pdd.generate_test.unfinished_prompt", _stub_unfinished)
    monkeypatch.setattr("pdd.generate_test.postprocess", _stub_postprocess)

    generate_test(
        prompt="Create calculator",
        code="def add(a, b): return a + b",
        example=None,
        strength=0.5,
        temperature=0.0,
        language="python",
        verbose=True  # Enable verbose output
    )

    captured = capsys.readouterr()
    assert "Traditional (code-based)" in captured.out, \
        "Verbose output should indicate traditional mode"


def test_generate_test_empty_string_example_treated_as_none():
    """
    Test that empty string for 'example' is treated as not provided.

    Issue 212: Empty strings should not count as "provided".
    """
    with pytest.raises(ValueError, match="Neither 'code' nor 'example' was provided"):
        generate_test(
            prompt="Test prompt",
            code="",  # Empty string
            example="",  # Empty string
            strength=0.5,
            temperature=0.0,
            language="python"
        )


def test_generate_test_whitespace_only_example_treated_as_none():
    """
    Test that whitespace-only 'example' is treated as not provided.

    Issue 212: Whitespace-only strings should not count as "provided".
    """
    with pytest.raises(ValueError, match="Neither 'code' nor 'example' was provided"):
        generate_test(
            prompt="Test prompt",
            code="   ",  # Whitespace only
            example="  \n  ",  # Whitespace only
            strength=0.5,
            temperature=0.0,
            language="python"
        )