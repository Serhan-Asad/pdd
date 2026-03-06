#!/bin/bash
cd /tmp/pdd_job_Pqv9H3sGy5Bg1ZLLBYPw_rzac9a76
pytest -x tests/test_agentic_architecture_orchestrator.py tests/test_issue_737_step_completion_markers.py::TestArchitectureOrchestratorStepMarkers -v 2>&1
