"""Unit tests for the xml_tagger module.

This module contains tests for verifying the XML tagging functionality
in the pdd.xml_tagger module.
"""

import pytest
from pdd.xml_tagger import xml_tagger

# Mock classes and functions
class MockXMLOutput:
    """Mock class for XMLOutput in testing."""
    def __init__(self, xml_tagged):
        self.xml_tagged = xml_tagged

    def to_dict(self):
        """Dummy method to avoid too-few-public-methods warning."""
        return {"xml_tagged": self.xml_tagged}

def mock_load_prompt_template(_):
    """Mock implementation of load_prompt_template."""
    return "mock_template"

def mock_llm_invoke(**kwargs):
    """Mock implementation of llm_invoke."""
    if kwargs.get('output_pydantic'):
        return {
            'result': MockXMLOutput(xml_tagged="<xml>mock tagged content</xml>"),
            'cost': 0.001,
            'model_name': 'mock-model'
        }
    return {
        'result': "<xml>mock analysis</xml>",
        'cost': 0.001,
        'model_name': 'mock-model'
    }

# Fixtures
@pytest.fixture
def mock_deps(monkeypatch):
    """Fixture to set up mock dependencies for tests."""
    monkeypatch.setattr('pdd.xml_tagger.load_prompt_template', mock_load_prompt_template)
    monkeypatch.setattr('pdd.xml_tagger.llm_invoke', mock_llm_invoke)

# Test cases
def test_successful_xml_tagging(mock_deps):
    """Test successful XML tagging with valid inputs."""
    raw_prompt = "Test prompt"
    result = xml_tagger(raw_prompt, strength=0.7, temperature=0.8)

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert isinstance(result[0], str)
    assert isinstance(result[1], float)
    assert isinstance(result[2], str)
    assert result[0] == "<xml>mock tagged content</xml>"
    assert result[1] > 0  # Total cost should be positive
    assert result[2] == "mock-model"

def test_verbose_output(mock_deps, capsys):
    """Test verbose output functionality."""
    raw_prompt = "Test prompt"
    xml_tagger(raw_prompt, strength=0.7, temperature=0.8, verbose=True)

    captured = capsys.readouterr()
    assert "Running XML conversion" in captured.out
    assert "Extracting final XML structure" in captured.out

def test_empty_prompt(mock_deps):
    """Test handling of empty prompt."""
    with pytest.raises(ValueError) as exc_info:
        xml_tagger("", strength=0.7, temperature=0.8)
    assert "raw_prompt must be a non-empty string" in str(exc_info.value)

def test_invalid_prompt_type(mock_deps):
    """Test handling of invalid prompt type."""
    with pytest.raises(ValueError) as exc_info:
        xml_tagger(None, strength=0.7, temperature=0.8)
    assert "raw_prompt must be a non-empty string" in str(exc_info.value)

def test_invalid_strength(mock_deps):
    """Test handling of invalid strength parameter."""
    with pytest.raises(ValueError) as exc_info:
        xml_tagger("Test prompt", strength=1.5, temperature=0.8)
    assert "strength must be between 0 and 1" in str(exc_info.value)

def test_invalid_temperature(mock_deps):
    """Test handling of invalid temperature parameter."""
    with pytest.raises(ValueError) as exc_info:
        xml_tagger("Test prompt", strength=0.7, temperature=-0.1)
    assert "temperature must be between 0 and 1" in str(exc_info.value)

def test_boundary_values(mock_deps):
    """Test boundary values for strength and temperature."""
    # Test minimum values
    result_min = xml_tagger("Test prompt", strength=0.0, temperature=0.0)
    assert isinstance(result_min, tuple)

    # Test maximum values
    result_max = xml_tagger("Test prompt", strength=1.0, temperature=1.0)
    assert isinstance(result_max, tuple)

@pytest.mark.parametrize("strength,temperature", [
    (0.3, 0.3),
    (0.5, 0.5),
    (0.7, 0.7),
])
def test_various_parameter_combinations(mock_deps, strength, temperature):
    """Test different combinations of valid strength and temperature values."""
    result = xml_tagger("Test prompt", strength=strength, temperature=temperature)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[1] > 0

def test_failed_template_loading(monkeypatch):
    """Test handling of failed template loading."""
    def mock_failed_template_load(_):
        return None

    monkeypatch.setattr('pdd.xml_tagger.load_prompt_template', mock_failed_template_load)

    with pytest.raises(ValueError) as exc_info:
        xml_tagger("Test prompt", strength=0.7, temperature=0.8)
    assert "Failed to load prompt templates" in str(exc_info.value)


# ── Tests for wrap_xml (issue #196) ──────────────────────────────────────────

class TestWrapXml:
    """Tests for the wrap_xml utility function.

    Issue #196: wrap_xml produces duplicate closing tags when given empty content.
    """

    def test_empty_content_no_duplicate_closing_tag(self):
        """Primary bug: empty content must produce <tag></tag>, not <tag></tag></tag>."""
        from pdd.xml_tagger import wrap_xml
        result = wrap_xml("test_tag", "")
        assert result == "<test_tag></test_tag>", f"Got: {result}"

    def test_non_empty_content_wrapping(self):
        """Non-empty content should be wrapped correctly."""
        from pdd.xml_tagger import wrap_xml
        result = wrap_xml("section", "hello world")
        assert result == "<section>hello world</section>"

    def test_content_with_newlines(self):
        """Content containing newlines should be preserved inside tags."""
        from pdd.xml_tagger import wrap_xml
        result = wrap_xml("block", "line1\nline2\nline3")
        assert result == "<block>line1\nline2\nline3</block>"

    def test_tag_with_special_characters(self):
        """Tag names with underscores and hyphens should work."""
        from pdd.xml_tagger import wrap_xml
        assert wrap_xml("my_tag", "content") == "<my_tag>content</my_tag>"
        assert wrap_xml("my-tag", "content") == "<my-tag>content</my-tag>"

    def test_whitespace_only_content(self):
        """Whitespace-only content should be wrapped, not treated as empty."""
        from pdd.xml_tagger import wrap_xml
        result = wrap_xml("tag", "   ")
        assert result == "<tag>   </tag>"

    def test_content_with_xml_characters(self):
        """Content containing XML-like characters should be wrapped as-is."""
        from pdd.xml_tagger import wrap_xml
        result = wrap_xml("outer", "<inner>value</inner>")
        assert result == "<outer><inner>value</inner></outer>"
