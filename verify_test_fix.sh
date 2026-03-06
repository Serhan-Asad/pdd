#!/bin/bash
cd /tmp/pdd_job_Pqv9H3sGy5Bg1ZLLBYPw_rzac9a76
pytest -x tests/test_agentic_test_orchestrator.py tests/test_issue_737_step_completion_markers.py::TestTestOrchestratorStepMarkers -v 2>&1
