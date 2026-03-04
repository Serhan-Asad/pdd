"""E2E test for sync pipeline validation boundary behavior.

Bug: The sync pipeline crashes when calling internal modules with default
parameter values. Specifically, temperature=0.0 (the default) triggers a
ValueError in the input validation of dependency resolution modules.

This manifests as a crash during `pdd sync` when the auto-deps step runs.
"""
import os
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def quiet_mode():
    with patch.dict(os.environ, {"PDD_QUIET": "1"}):
        yield


class TestSyncPipelineValidationBoundaries:
    """E2E tests verifying the sync pipeline handles boundary values correctly."""

    def test_dependency_resolver_accepts_zero_temperature(self):
        """The dependency resolver must accept temperature=0.0 (its default)."""
        from pdd.auto_include import _validate_input
        # temperature=0.0 is the default — must not raise
        _validate_input("some prompt", "/some/dir", strength=0.5, temperature=0.0)

    def test_dependency_resolver_accepts_zero_strength(self):
        """The dependency resolver must accept strength=0.0."""
        from pdd.auto_include import _validate_input
        _validate_input("some prompt", "/some/dir", strength=0.0, temperature=0.5)

    def test_dependency_resolver_accepts_both_zero(self):
        """Both strength=0.0 and temperature=0.0 must be accepted."""
        from pdd.auto_include import _validate_input
        _validate_input("some prompt", "/some/dir", strength=0.0, temperature=0.0)

    def test_sync_pipeline_default_temperature_does_not_crash(self):
        """Simulates sync pipeline calling auto_include with default temperature."""
        from pdd.auto_include import auto_include
        with patch('pdd.auto_include._load_prompts', return_value=("p1", "p2")):
            with patch('pdd.auto_include._summarize', return_value=("csv", 0.0, "model")):
                with patch('pdd.auto_include._get_available_includes_from_csv', return_value=[]):
                    with patch('pdd.auto_include._run_llm_and_extract', return_value=("deps", 0.0, "model")):
                        with patch('pdd.auto_include._filter_existing_includes', return_value=""):
                            with patch('pdd.auto_include._filter_self_references', return_value=""):
                                with patch('pdd.auto_include._fix_malformed_includes', return_value=""):
                                    with patch('pdd.auto_include._detect_circular_dependencies', return_value=[]):
                                        result = auto_include(
                                            input_prompt="Test prompt",
                                            directory_path="/valid/dir",
                                            strength=0.5,
                                            # temperature defaults to 0.0
                                        )
                                        assert isinstance(result, tuple)
