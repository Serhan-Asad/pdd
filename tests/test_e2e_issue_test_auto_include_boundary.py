"""E2E test for auto_include validation boundary behavior.

Bug: auto_include._validate_input() rejects strength=0.0 and temperature=0.0
with ValueError even though 0.0 is a valid lower bound (the valid range is
[0, 1] inclusive). This manifests as an e2e failure when pdd sync calls
auto_include with default temperature=0.0 — the sync pipeline crashes with
"Temperature must be between 0 and 1" instead of proceeding.

This tests the full auto_include pipeline entry point, not just the validator,
to verify the e2e behavior is correct.
"""
import os
import pytest
from unittest.mock import patch, MagicMock

from pdd.auto_include import auto_include, _validate_input


@pytest.fixture(autouse=True)
def quiet_mode():
    with patch.dict(os.environ, {"PDD_QUIET": "1"}):
        yield


class TestAutoIncludeValidationBoundaries:
    """E2E tests for auto_include input validation at boundaries."""

    def test_validate_input_accepts_strength_zero(self):
        """strength=0.0 is the lower bound and must be accepted."""
        # Should NOT raise
        _validate_input("some prompt", "/some/dir", strength=0.0, temperature=0.5)

    def test_validate_input_accepts_temperature_zero(self):
        """temperature=0.0 is the lower bound and must be accepted."""
        # Should NOT raise
        _validate_input("some prompt", "/some/dir", strength=0.5, temperature=0.0)

    def test_validate_input_accepts_both_zero(self):
        """Both strength=0.0 and temperature=0.0 should be accepted together."""
        _validate_input("some prompt", "/some/dir", strength=0.0, temperature=0.0)

    def test_auto_include_default_temperature_zero(self):
        """auto_include default temperature is 0.0 — this must not crash."""
        with patch('pdd.auto_include._load_prompts', return_value=("prompt1", "prompt2")):
            with patch('pdd.auto_include._summarize', return_value=("csv", 0.0, "model")):
                with patch('pdd.auto_include._get_available_includes_from_csv', return_value=[]):
                    with patch('pdd.auto_include._run_llm_and_extract', return_value=("deps", 0.0, "model")):
                        with patch('pdd.auto_include._filter_existing_includes', return_value=""):
                            with patch('pdd.auto_include._filter_self_references', return_value=""):
                                with patch('pdd.auto_include._fix_malformed_includes', return_value=""):
                                    with patch('pdd.auto_include._detect_circular_dependencies', return_value=[]):
                                        # Default temperature=0.0 should not raise
                                        result = auto_include(
                                            input_prompt="Test prompt content",
                                            directory_path="/some/valid/dir",
                                            strength=0.5,
                                            # temperature defaults to 0.0
                                        )
                                        assert isinstance(result, tuple)
                                        assert len(result) == 4

    def test_validate_rejects_negative_strength(self):
        """Negative strength should still be rejected."""
        with pytest.raises(ValueError, match="Strength must be between 0 and 1"):
            _validate_input("prompt", "/dir", strength=-0.1, temperature=0.5)

    def test_validate_rejects_strength_above_one(self):
        """Strength > 1 should still be rejected."""
        with pytest.raises(ValueError, match="Strength must be between 0 and 1"):
            _validate_input("prompt", "/dir", strength=1.1, temperature=0.5)
