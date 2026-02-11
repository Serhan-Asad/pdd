"""
E2E tests for issue #85: /api/v1/prompts/models returns HTTP 500
when PDD_MODEL_DEFAULT is not set.

These tests exercise the full HTTP path via FastAPI TestClient,
unlike the unit tests which call the handler function directly.
"""

import sys
import types
import pytest
import pandas as pd
from unittest.mock import MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def e2e_app():
    """
    Create a FastAPI app with the real prompts router mounted,
    mocking only external dependencies (security, llm_invoke, etc.)
    but NOT the buggy component itself.
    """
    # Save original modules
    original_modules = {}
    modules_to_mock = [
        "pdd.server.security",
        "pdd.server.token_counter",
        "pdd.preprocess",
        "pdd.sync_determine_operation",
        "pdd.llm_invoke",
    ]
    modules_to_clear = [
        "pdd.server.routes.prompts",
        "pdd.server.routes",
    ]

    for mod in modules_to_mock + modules_to_clear:
        if mod in sys.modules:
            original_modules[mod] = sys.modules[mod]

    # Mock external dependencies
    mock_security = types.ModuleType("pdd.server.security")
    mock_security.SecurityError = type("SecurityError", (Exception,), {"__init__": lambda self, msg: Exception.__init__(self, msg)})
    mock_security.PathValidator = MagicMock()
    mock_security.configure_cors = MagicMock()
    mock_security.create_token_dependency = MagicMock()
    mock_security.SecurityLoggingMiddleware = MagicMock()
    mock_security.DEFAULT_BLACKLIST = []
    sys.modules["pdd.server.security"] = mock_security

    mock_token_counter = types.ModuleType("pdd.server.token_counter")
    mock_token_counter.get_token_metrics = MagicMock()
    mock_token_counter.MODEL_CONTEXT_LIMITS = {"default": 128000}
    sys.modules["pdd.server.token_counter"] = mock_token_counter

    mock_preprocess = types.ModuleType("pdd.preprocess")
    mock_preprocess.preprocess = MagicMock()
    sys.modules["pdd.preprocess"] = mock_preprocess

    mock_sync_op = types.ModuleType("pdd.sync_determine_operation")
    mock_sync_op.read_fingerprint = MagicMock()
    mock_sync_op.get_pdd_file_paths = MagicMock()
    mock_sync_op.calculate_sha256 = MagicMock()
    sys.modules["pdd.sync_determine_operation"] = mock_sync_op

    # Mock llm_invoke with DEFAULT_BASE_MODEL = None (the bug condition)
    mock_llm_invoke = types.ModuleType("pdd.llm_invoke")
    mock_llm_invoke.llm_invoke = MagicMock()
    mock_llm_invoke._load_model_data = MagicMock(return_value=pd.DataFrame([
        {
            "model": "claude-sonnet-4-20250514",
            "provider": "Anthropic",
            "input": 3.0,
            "output": 15.0,
            "coding_arena_elo": 1400,
            "max_reasoning_tokens": 0,
            "reasoning_type": "none",
            "structured_output": True,
        },
    ]))
    mock_llm_invoke.LLM_MODEL_CSV_PATH = "/mock/llm_model.csv"
    # THIS IS THE BUG: DEFAULT_BASE_MODEL is None when PDD_MODEL_DEFAULT is unset
    mock_llm_invoke.DEFAULT_BASE_MODEL = None
    sys.modules["pdd.llm_invoke"] = mock_llm_invoke

    # Clear cached imports
    for mod_name in list(sys.modules.keys()):
        if "pdd.server.routes.prompts" in mod_name:
            del sys.modules[mod_name]

    # Import the real router
    from pdd.server.routes.prompts import router

    # Build a minimal FastAPI app with the real router
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    yield client

    # Cleanup
    for mod in modules_to_mock:
        if mod in sys.modules:
            del sys.modules[mod]
    for mod_name in list(sys.modules.keys()):
        if "pdd.server.routes.prompts" in mod_name:
            del sys.modules[mod_name]
    for mod, original in original_modules.items():
        sys.modules[mod] = original


def test_models_endpoint_returns_200_when_default_model_none(e2e_app):
    """
    E2E test for issue #85: GET /api/v1/prompts/models should return 200
    even when PDD_MODEL_DEFAULT is not set (DEFAULT_BASE_MODEL is None).

    Currently this returns HTTP 500 due to Pydantic ValidationError.
    """
    response = e2e_app.get("/api/v1/prompts/models")

    # BUG: This currently returns 500 instead of 200
    assert response.status_code == 200, (
        f"Expected 200 but got {response.status_code}. "
        f"Response: {response.json()}"
    )
    data = response.json()
    assert "models" in data
    assert "default_model" in data
    # default_model should be a string, not None
    assert isinstance(data["default_model"], str)


def test_models_endpoint_error_message_on_500(e2e_app):
    """
    E2E test verifying the exact error users see: the Pydantic ValidationError
    in the HTTP 500 response body mentioning 'default_model'.
    """
    response = e2e_app.get("/api/v1/prompts/models")

    if response.status_code == 500:
        detail = response.json().get("detail", "")
        # Confirm this is the specific bug, not some other 500
        assert "default_model" in detail, (
            f"Got 500 but not the expected ValidationError. Detail: {detail}"
        )
        # This assertion always fails on buggy code — the test "passes"
        # by detecting the bug
        pytest.fail(
            f"Bug #85 confirmed: endpoint returns 500 with ValidationError: {detail}"
        )
