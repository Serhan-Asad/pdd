#!/bin/bash
pytest -vv tests/test_issue_737_step_completion_markers.py::TestTestOrchestratorStepMarkers tests/test_agentic_test_orchestrator.py 2>&1
