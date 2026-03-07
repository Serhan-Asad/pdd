# tests/test_stop_condition_parser.py
import pytest

try:
    from pdd.agentic_common import extract_stop_condition
except ImportError:
    # Dummy to fail the tests cleanly until implemented
    def extract_stop_condition(output: str) -> str:
        return None

def test_extract_stop_condition_valid_tag():
    """Test 1: Intentional Hard Stop via XML Tag (Data Flow)"""
    output = "Here is some text.\n<pdd_stop_condition>Clarification Needed</pdd_stop_condition>\nMore text."
    assert extract_stop_condition(output) == "Clarification Needed"

def test_extract_stop_condition_case_and_whitespace():
    """Test 4: Case-Insensitivity and Formatting Edge Cases"""
    output = "<pdd_stop_condition>  architectural decision needed  </pdd_stop_condition>"
    assert extract_stop_condition(output) == "Architectural decision needed"

def test_extract_stop_condition_colloquial_ignore():
    """Test 3: Ignore Colloquial Mentions / False Positives"""
    output = "I think Clarification Needed is a good idea but no tags here."
    assert extract_stop_condition(output) is None
