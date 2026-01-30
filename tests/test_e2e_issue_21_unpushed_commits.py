"""
E2E Test for Issue #21: Agentic fix doesn't push commits when exiting early at Step 2

This test reproduces the bug where commits created by the LLM agent during Step 1
are not pushed to the remote when the workflow exits early at Step 2 (ALL_TESTS_PASS).

The Bug:
--------
File: pdd/agentic_e2e_fix_orchestrator.py
Function: _commit_and_push() (lines 198-273)

When the agentic e2e fix workflow exits early at Step 2:
1. initial_file_hashes = _get_file_hashes(cwd) returns {} (no uncommitted files)
2. LLM agent in Step 1 modifies files and creates a commit
3. Step 2 detects ALL_TESTS_PASS, workflow exits
4. _commit_and_push() is called at line 521
5. current_hashes = _get_file_hashes(cwd) returns {} (working tree is clean)
6. Function returns early at lines 237-238 with "No changes to commit"
7. git push is NEVER executed (lines 263-268)

Expected Behavior:
-----------------
When the workflow completes successfully (success=True), ALL commits created during
the workflow should be pushed to the remote, regardless of whether the workflow
exits early or runs all 9 steps.

This Test:
----------
1. Creates a real git repository with a remote
2. Mocks run_agentic_task to simulate:
   - Step 1: Creates a commit (like the LLM agent would)
   - Step 2: Returns ALL_TESTS_PASS (triggers early exit)
3. Calls run_agentic_e2e_fix_orchestrator
4. Verifies the commit exists locally
5. **Verifies the commit was NOT pushed to remote** (the bug)

The test should FAIL on buggy code and PASS once the fix is applied.
"""

import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os


@pytest.fixture
def git_repo_with_remote(tmp_path):
    """
    Create a real git repository with a remote for testing commit/push behavior.

    This fixture creates:
    - A "remote" bare repository (simulating GitHub)
    - A "local" working repository (where the orchestrator runs)
    - A branch "fix/issue-21" tracking the remote

    Returns the local repo path.
    """
    # Create remote (bare) repository
    remote_dir = tmp_path / "remote.git"
    remote_dir.mkdir()
    subprocess.run(["git", "init", "--bare"], cwd=remote_dir, check=True, capture_output=True)

    # Create local repository
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=local_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=local_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=local_dir, check=True, capture_output=True)

    # Create initial commit on main branch
    (local_dir / "README.md").write_text("# Test Repository")
    subprocess.run(["git", "add", "README.md"], cwd=local_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=local_dir, check=True, capture_output=True)

    # Set up remote and push main
    subprocess.run(["git", "remote", "add", "origin", str(remote_dir)], cwd=local_dir, check=True, capture_output=True)
    subprocess.run(["git", "push", "-u", "origin", "main"], cwd=local_dir, check=True, capture_output=True)

    # Create and push fix/issue-21 branch (simulating what `pdd bug` does)
    subprocess.run(["git", "checkout", "-b", "fix/issue-21"], cwd=local_dir, check=True, capture_output=True)
    (local_dir / "test_file.txt").write_text("Initial test content")
    subprocess.run(["git", "add", "test_file.txt"], cwd=local_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Add failing tests for issue #21"], cwd=local_dir, check=True, capture_output=True)
    subprocess.run(["git", "push", "-u", "origin", "fix/issue-21"], cwd=local_dir, check=True, capture_output=True)

    # Create .pdd directory (required by orchestrator)
    (local_dir / ".pdd").mkdir()

    return local_dir


def get_local_commits(repo_dir, branch="fix/issue-21"):
    """Get list of commit SHAs on local branch."""
    result = subprocess.run(
        ["git", "log", branch, "--format=%H"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip().split('\n')


def get_remote_commits(repo_dir, remote_branch="origin/fix/issue-21"):
    """Get list of commit SHAs on remote branch."""
    # Fetch latest from remote
    subprocess.run(["git", "fetch", "origin"], cwd=repo_dir, capture_output=True, check=True)
    result = subprocess.run(
        ["git", "log", remote_branch, "--format=%H"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip().split('\n')


def count_unpushed_commits(repo_dir, branch="fix/issue-21", remote="origin/fix/issue-21"):
    """Count commits that exist locally but not on remote."""
    result = subprocess.run(
        ["git", "rev-list", "--count", f"{remote}..{branch}"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True
    )
    return int(result.stdout.strip())


class TestIssue21UnpushedCommitsE2E:
    """
    E2E tests for Issue #21: Verify commits are pushed when workflow exits early.

    These tests exercise the orchestrator's commit and push logic with real git
    operations to detect when commits created by LLM agents are not pushed.
    """

    def test_early_exit_step2_does_not_push_commits(self, git_repo_with_remote, monkeypatch):
        """
        E2E Test: Early exit at Step 2 should push commits created in Step 1 (BUG: it doesn't).

        This test reproduces the exact bug scenario:
        1. Workflow starts with clean working tree (initial_file_hashes = {})
        2. Step 1: LLM agent modifies a file and creates a commit
        3. Step 2: Returns ALL_TESTS_PASS, workflow exits early
        4. _commit_and_push() is called
        5. BUG: Returns "No changes to commit" without pushing the Step 1 commit

        Expected behavior (after fix):
        - The Step 1 commit should be pushed to remote
        - unpushed_commits should be 0

        Actual behavior (before fix):
        - The Step 1 commit is NOT pushed
        - unpushed_commits is 1
        - This test FAILS, confirming the bug
        """
        repo_dir = git_repo_with_remote

        # Set environment to avoid external dependencies
        monkeypatch.setenv("PDD_FORCE_LOCAL", "1")

        # Track which steps were called
        steps_called = []

        def mock_run_agentic_task(instruction, cwd, verbose, quiet, timeout, label, max_retries):
            """
            Mock LLM interaction that simulates the bug scenario.

            Step 1: Create a commit (simulating LLM agent fixing the code)
            Step 2: Return ALL_TESTS_PASS (triggers early exit)

            IMPORTANT: This mock creates a REAL commit in Step 1, just like the
            actual LLM agent would. This commit should be pushed at workflow end.
            """
            import re
            match = re.search(r"step(\d+)", label)
            if match:
                step_num = int(match.group(1))
                steps_called.append(step_num)

                if step_num == 1:
                    # Simulate LLM agent fixing the code and creating a commit
                    # This is what the REAL agent does via run_agentic_task
                    test_file = cwd / "test_file.txt"
                    test_file.write_text("Fixed content from Step 1")

                    # Stage and commit (simulating what the agent does)
                    subprocess.run(["git", "add", "test_file.txt"], cwd=cwd, check=True, capture_output=True)
                    subprocess.run(
                        ["git", "commit", "-m", "Fix from LLM agent in Step 1"],
                        cwd=cwd,
                        check=True,
                        capture_output=True
                    )

                    return (True, "Fixed the issue by modifying test_file.txt", 0.01, "mock-model")

                elif step_num == 2:
                    # Return ALL_TESTS_PASS to trigger early exit
                    return (True, "ALL_TESTS_PASS - All tests are passing", 0.01, "mock-model")

            return (True, f"Mock success for {label}", 0.01, "mock-model")

        # Mock state management to avoid side effects
        def mock_save_state(*args, **kwargs):
            return None

        def mock_load_state(*args, **kwargs):
            return None, None

        def mock_clear_state(*args, **kwargs):
            pass

        # Get initial commit counts
        initial_local_commits = get_local_commits(repo_dir)
        initial_remote_commits = get_remote_commits(repo_dir)
        initial_unpushed = count_unpushed_commits(repo_dir)

        # Patch and run the orchestrator
        with patch('pdd.agentic_e2e_fix_orchestrator.run_agentic_task', side_effect=mock_run_agentic_task):
            with patch('pdd.agentic_e2e_fix_orchestrator.save_workflow_state', side_effect=mock_save_state):
                with patch('pdd.agentic_e2e_fix_orchestrator.load_workflow_state', side_effect=mock_load_state):
                    with patch('pdd.agentic_e2e_fix_orchestrator.clear_workflow_state', side_effect=mock_clear_state):
                        from pdd.agentic_e2e_fix_orchestrator import run_agentic_e2e_fix_orchestrator

                        success, message, cost, model, files = run_agentic_e2e_fix_orchestrator(
                            issue_url="https://github.com/test/repo/issues/21",
                            issue_content="Test issue for bug #21",
                            repo_owner="test",
                            repo_name="repo",
                            issue_number=21,
                            issue_author="test-user",
                            issue_title="Agentic fix doesn't push commits when exiting early",
                            cwd=repo_dir,
                            max_cycles=1,
                            resume=False,
                            verbose=False,
                            quiet=True,
                            use_github_state=False,
                            protect_tests=False
                        )

        # Verify workflow completed successfully
        assert success is True, f"Workflow should succeed, but got success={success}, message={message}"
        assert 1 in steps_called, f"Step 1 should have been called. Steps called: {steps_called}"
        assert 2 in steps_called, f"Step 2 should have been called. Steps called: {steps_called}"

        # Verify a new commit was created locally
        final_local_commits = get_local_commits(repo_dir)
        assert len(final_local_commits) > len(initial_local_commits), \
            "A new commit should have been created in Step 1"

        # BUG CHECK: Verify the commit was pushed to remote
        final_remote_commits = get_remote_commits(repo_dir)
        final_unpushed = count_unpushed_commits(repo_dir)

        # This assertion FAILS on buggy code (final_unpushed will be 1)
        # It PASSES after the fix (final_unpushed will be 0)
        assert final_unpushed == 0, (
            f"BUG DETECTED (Issue #21): Commit created in Step 1 was NOT pushed to remote!\n\n"
            f"Initial state:\n"
            f"  - Local commits: {len(initial_local_commits)}\n"
            f"  - Remote commits: {len(initial_remote_commits)}\n"
            f"  - Unpushed: {initial_unpushed}\n\n"
            f"Final state:\n"
            f"  - Local commits: {len(final_local_commits)} (+{len(final_local_commits) - len(initial_local_commits)})\n"
            f"  - Remote commits: {len(final_remote_commits)} (+{len(final_remote_commits) - len(initial_remote_commits)})\n"
            f"  - Unpushed: {final_unpushed} ❌\n\n"
            f"Expected: Unpushed commits = 0 (all commits pushed)\n"
            f"Actual: Unpushed commits = {final_unpushed}\n\n"
            f"Root cause: _commit_and_push() at line 237-238 returns early with 'No changes to commit'\n"
            f"without executing 'git push' because _get_file_hashes() only detects uncommitted files,\n"
            f"but the LLM agent already committed the changes in Step 1."
        )


class TestIssue21RegressionPrevention:
    """
    Regression tests to ensure the fix doesn't break other scenarios.
    """

    def test_full_workflow_with_uncommitted_changes_still_works(self, git_repo_with_remote, monkeypatch):
        """
        Regression Test: Full workflow (9 steps) with uncommitted changes should work.

        This test verifies that the fix for early exit doesn't break the normal case
        where the workflow completes all 9 steps and has uncommitted changes to commit.
        """
        repo_dir = git_repo_with_remote
        monkeypatch.setenv("PDD_FORCE_LOCAL", "1")

        steps_called = []

        def mock_run_agentic_task(instruction, cwd, verbose, quiet, timeout, label, max_retries):
            import re
            match = re.search(r"step(\d+)", label)
            if match:
                step_num = int(match.group(1))
                steps_called.append(step_num)

                # Step 8: Create uncommitted changes (NOT a commit)
                if step_num == 8:
                    test_file = cwd / "new_file.txt"
                    test_file.write_text("Uncommitted change from Step 8")
                    return (True, "Fixed the code", 0.01, "mock-model")

                # Step 9: Return MAX_CYCLES_REACHED to end workflow
                if step_num == 9:
                    return (True, "MAX_CYCLES_REACHED - out of cycles", 0.01, "mock-model")

            return (True, f"Mock success for {label}", 0.01, "mock-model")

        with patch('pdd.agentic_e2e_fix_orchestrator.run_agentic_task', side_effect=mock_run_agentic_task):
            with patch('pdd.agentic_e2e_fix_orchestrator.save_workflow_state', return_value=None):
                with patch('pdd.agentic_e2e_fix_orchestrator.load_workflow_state', return_value=(None, None)):
                    with patch('pdd.agentic_e2e_fix_orchestrator.clear_workflow_state', return_value=None):
                        from pdd.agentic_e2e_fix_orchestrator import run_agentic_e2e_fix_orchestrator

                        initial_unpushed = count_unpushed_commits(repo_dir)

                        success, message, cost, model, files = run_agentic_e2e_fix_orchestrator(
                            issue_url="https://github.com/test/repo/issues/21",
                            issue_content="Test issue",
                            repo_owner="test",
                            repo_name="repo",
                            issue_number=21,
                            issue_author="test-user",
                            issue_title="Test issue",
                            cwd=repo_dir,
                            max_cycles=1,
                            resume=False,
                            verbose=False,
                            quiet=True,
                            use_github_state=False,
                            protect_tests=False
                        )

        # Verify workflow ran
        assert success is False, "Workflow should fail (MAX_CYCLES_REACHED)"

        # Verify the uncommitted change was committed and pushed
        final_unpushed = count_unpushed_commits(repo_dir)
        assert final_unpushed == 0, \
            f"Uncommitted changes should be committed and pushed (unpushed={final_unpushed})"

    def test_no_changes_and_no_unpushed_commits(self, git_repo_with_remote, monkeypatch):
        """
        Regression Test: Workflow with no changes and no unpushed commits should succeed.

        This is the true negative case - no work was done, nothing to commit or push.
        """
        repo_dir = git_repo_with_remote
        monkeypatch.setenv("PDD_FORCE_LOCAL", "1")

        def mock_run_agentic_task(instruction, cwd, verbose, quiet, timeout, label, max_retries):
            # Don't create any changes or commits
            import re
            match = re.search(r"step(\d+)", label)
            if match:
                step_num = int(match.group(1))
                if step_num == 2:
                    return (True, "ALL_TESTS_PASS - no changes needed", 0.01, "mock-model")
            return (True, f"Mock success for {label}", 0.01, "mock-model")

        with patch('pdd.agentic_e2e_fix_orchestrator.run_agentic_task', side_effect=mock_run_agentic_task):
            with patch('pdd.agentic_e2e_fix_orchestrator.save_workflow_state', return_value=None):
                with patch('pdd.agentic_e2e_fix_orchestrator.load_workflow_state', return_value=(None, None)):
                    with patch('pdd.agentic_e2e_fix_orchestrator.clear_workflow_state', return_value=None):
                        from pdd.agentic_e2e_fix_orchestrator import run_agentic_e2e_fix_orchestrator

                        initial_unpushed = count_unpushed_commits(repo_dir)

                        success, message, cost, model, files = run_agentic_e2e_fix_orchestrator(
                            issue_url="https://github.com/test/repo/issues/21",
                            issue_content="Test issue",
                            repo_owner="test",
                            repo_name="repo",
                            issue_number=21,
                            issue_author="test-user",
                            issue_title="Test issue",
                            cwd=repo_dir,
                            max_cycles=1,
                            resume=False,
                            verbose=False,
                            quiet=True,
                            use_github_state=False,
                            protect_tests=False
                        )

        # Verify workflow succeeded
        assert success is True, f"Workflow should succeed with no changes"

        # Verify no new unpushed commits
        final_unpushed = count_unpushed_commits(repo_dir)
        assert final_unpushed == initial_unpushed, \
            f"No new unpushed commits should be created (initial={initial_unpushed}, final={final_unpushed})"
