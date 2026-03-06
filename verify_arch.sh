#!/bin/bash
pytest -vv tests/test_issue_737_step_completion_markers.py::TestArchitectureOrchestratorStepMarkers tests/test_agentic_architecture_orchestrator.py 2>&1
