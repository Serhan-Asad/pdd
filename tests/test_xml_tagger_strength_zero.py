"""Unit test for xml_tagger strength=0.0 boundary.

Bug: xml_tagger rejects strength=0.0 with ValueError even though 0.0 is a valid
strength value (the valid range is [0, 1] inclusive).
"""
import pytest
from unittest.mock import MagicMock


class MockXMLOutput:
    def __init__(self, xml_tagged):
        self.xml_tagged = xml_tagged


def mock_load_prompt_template(_):
    return "mock_template"


def mock_llm_invoke(**kwargs):
    if kwargs.get('output_pydantic'):
        return {
            'result': MockXMLOutput(xml_tagged="<xml>tagged</xml>"),
            'cost': 0.001,
            'model_name': 'mock-model'
        }
    return {
        'result': "<xml>analysis</xml>",
        'cost': 0.001,
        'model_name': 'mock-model'
    }


@pytest.fixture
def mock_deps(monkeypatch):
    monkeypatch.setattr('pdd.xml_tagger.load_prompt_template', mock_load_prompt_template)
    monkeypatch.setattr('pdd.xml_tagger.llm_invoke', mock_llm_invoke)


def test_xml_tagger_accepts_strength_zero(mock_deps):
    """strength=0.0 is the lower bound and should be accepted."""
    from pdd.xml_tagger import xml_tagger
    result = xml_tagger("Test prompt", strength=0.0, temperature=0.5)
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_xml_tagger_accepts_strength_one(mock_deps):
    """strength=1.0 is the upper bound and should be accepted."""
    from pdd.xml_tagger import xml_tagger
    result = xml_tagger("Test prompt", strength=1.0, temperature=0.5)
    assert isinstance(result, tuple)
    assert len(result) == 3
