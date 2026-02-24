from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

from rich.console import Console

from .agentic_common import (
    run_agentic_task,
    load_workflow_state,
    save_workflow_state,
    clear_workflow_state,
    DEFAULT_MAX_RETRIES
)
from .load_prompt_template import load_prompt_template
from .preprocess import preprocess

# Initialize console
console = Console()

# Per-step timeouts for the 11-step agentic bug workflow
BUG_STEP_TIMEOUTS: Dict[Union[int, float], float] = {
    1: 240.0,    # Duplicate Check
    2: 400.0,    # Docs Check
    3: 400.0,    # Triage
    4: 600.0,    # Reproduce (Complex)
    5: 600.0,    # Root Cause (Complex)
    5.5: 600.0,  # Prompt Classification (may auto-fix prompts)
    6: 340.0,    # Test Plan
    7: 1000.0,   # Generate Unit Test (Most Complex)
    8: 600.0,    # Verify Unit Test
    9: 2000.0,   # E2E Test (Complex - needs to discover env & run tests)
    10: 240.0,   # Create PR
}


def _get_git_root(cwd: Path) -> Optional[Path]:
    """Get the root directory of the git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        return None


def _get_state_dir(cwd: Path) -> Path:
    """Return path to state directory relative to git root."""
    root = _get_git_root(cwd) or cwd
    return root / ".pdd" / "bug-state"


def _worktree_exists(cwd: Path, worktree_path: Path) -> bool:
    """Check if a path is a registered git worktree."""
    try:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return str(worktree_path.resolve()) in result.stdout
    except subprocess.CalledProcessError:
        return False


def _branch_exists(cwd: Path, branch: str) -> bool:
    """Check if a local git branch exists."""
    try:
        subprocess.run(
            ["git", "show-ref", "--verify", f"refs/heads/{branch}"],
            cwd=cwd,
            capture_output=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _remove_worktree(cwd: Path, worktree_path: Path) -> Tuple[bool, str]:
    """Remove a git worktree."""
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_path)],
            cwd=cwd,
            capture_output=True,
            check=True
        )
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, e.stderr.decode('utf-8')


def _delete_branch(cwd: Path, branch: str) -> Tuple[bool, str]:
    """Force delete a git branch."""
    try:
        subprocess.run(
            ["git", "branch", "-D", branch],
            cwd=cwd,
            capture_output=True,
            check=True
        )
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, e.stderr.decode('utf-8')


def _setup_worktree(cwd: Path, issue_number: int, quiet: bool, resume_existing: bool = False) -> Tuple[Optional[Path], Optional[str]]:
    """
    Sets up an isolated git worktree for the issue fix.

    Args:
        cwd: Current working directory
        issue_number: GitHub issue number
        quiet: Suppress output
        resume_existing: If True, keep existing branch with accumulated work

    Returns (worktree_path, error_message).
    """
    git_root = _get_git_root(cwd)
    if not git_root:
        return None, "Current directory is not a git repository."

    worktree_rel_path = Path(".pdd") / "worktrees" / f"fix-issue-{issue_number}"
    worktree_path = git_root / worktree_rel_path
    branch_name = f"fix/issue-{issue_number}"

    # 1. Clean up existing worktree at path (always needed to create fresh worktree)
    if worktree_path.exists():
        if _worktree_exists(git_root, worktree_path):
            if not quiet:
                console.print(f"[yellow]Removing existing worktree at {worktree_path}[/yellow]")
            success, err = _remove_worktree(git_root, worktree_path)
            if not success:
                return None, f"Failed to remove existing worktree: {err}"
        else:
            if not quiet:
                console.print(f"[yellow]Removing stale directory at {worktree_path}[/yellow]")
            shutil.rmtree(worktree_path)

    # 2. Handle existing branch based on resume_existing
    branch_exists = _branch_exists(git_root, branch_name)

    if branch_exists:
        if resume_existing:
            if not quiet:
                console.print(f"[blue]Resuming with existing branch: {branch_name}[/blue]")
        else:
            if not quiet:
                console.print(f"[yellow]Removing existing branch {branch_name}[/yellow]")
            success, err = _delete_branch(git_root, branch_name)
            if not success:
                return None, f"Failed to delete existing branch: {err}"

    # 3. Create worktree
    try:
        worktree_path.parent.mkdir(parents=True, exist_ok=True)

        if branch_exists and resume_existing:
            subprocess.run(
                ["git", "worktree", "add", str(worktree_path), branch_name],
                cwd=git_root,
                capture_output=True,
                check=True
            )
        else:
            subprocess.run(
                ["git", "worktree", "add", "-b", branch_name, str(worktree_path), "HEAD"],
                cwd=git_root,
                capture_output=True,
                check=True
            )
        return worktree_path, None
    except subprocess.CalledProcessError as e:
        return None, f"Failed to create worktree: {e.stderr.decode('utf-8')}"


def run_agentic_bug_orchestrator(
    issue_url: str,
    issue_content: str,
    repo_owner: str,
    repo_name: str,
    issue_number: int,
    issue_author: str,
    issue_title: str,
    *,
    cwd: Path,
    verbose: bool = False,
    quiet: bool = False,
    timeout_adder: float = 0.0,
    use_github_state: bool = True
) -> Tuple[bool, str, float, str, List[str]]:
    """
    Orchestrates the 11-step agentic bug investigation workflow.

    Returns:
        (success, final_message, total_cost, model_used, changed_files)
    """
    
    if not quiet:
        console.print(f"🔍 Investigating issue #{issue_number}: \"{issue_title}\"")

    # Context accumulation
    context: Dict[str, Any] = {
        "issue_url": issue_url,
        "issue_content": issue_content,
        "repo_owner": repo_owner,
        "repo_name": repo_name,
        "issue_number": issue_number,
        "issue_author": issue_author,
        "issue_title": issue_title,
    }

    total_cost = 0.0
    last_model_used = "unknown"
    changed_files: List[str] = []
    current_cwd = cwd
    worktree_path: Optional[Path] = None
    github_comment_id: Optional[int] = None

    # Resume: Load existing state if available
    state_dir = _get_state_dir(cwd)
    state, loaded_gh_id = load_workflow_state(
        cwd=cwd,
        issue_number=issue_number,
        workflow_type="bug",
        state_dir=state_dir,
        repo_owner=repo_owner,
        repo_name=repo_name,
        use_github_state=use_github_state
    )

    step_outputs: Dict[str, str] = {}
    last_completed_step = 0

    if state is not None:
        last_completed_step = state.get("last_completed_step", 0)
        cached_outputs = state.get("step_outputs", {})

        # Validate cached state — find actual last successful step
        if cached_outputs:
            step_order = [1, 2, 3, 4, 5, 5.5, 6, 7, 8, 9, 10]
            actual_last_success: Union[int, float] = 0
            for sn in step_order:
                output_val = cached_outputs.get(str(sn), "")
                if not output_val:
                    break
                if output_val.startswith("FAILED:"):
                    break
                actual_last_success = sn
            if actual_last_success < last_completed_step:
                if not quiet:
                    console.print(f"[yellow]State validation: correcting last_completed_step from {last_completed_step} to {actual_last_success} (found FAILED steps in cache)[/yellow]")
                last_completed_step = actual_last_success

        if last_completed_step == 5:
            resume_start_step: Union[int, float] = 5.5
        elif last_completed_step == 5.5:
            resume_start_step = 6
        else:
            resume_start_step = last_completed_step + 1
        if not quiet:
            console.print(f"[yellow]Resuming from step {resume_start_step} (steps 1-{last_completed_step} cached)[/yellow]")

        total_cost = state.get("total_cost", 0.0)
        last_model_used = state.get("model_used", "unknown")
        step_outputs = state.get("step_outputs", {})
        changed_files = state.get("changed_files", [])
        github_comment_id = loaded_gh_id

        wt_path_str = state.get("worktree_path")
        if wt_path_str:
            worktree_path = Path(wt_path_str)
            if worktree_path.exists():
                current_cwd = worktree_path
            else:
                wt_path, err = _setup_worktree(cwd, issue_number, quiet, resume_existing=True)
                if err:
                    return False, f"Failed to recreate worktree on resume: {err}", total_cost, last_model_used, []
                worktree_path = wt_path
                current_cwd = worktree_path
            context["worktree_path"] = str(worktree_path)

        for step_key, output in step_outputs.items():
            fixed_key = str(step_key).replace(".", "_")
            escaped_output = output.replace("{", "{{").replace("}", "}}").replace("\n", "\\n")
            context[f"step{fixed_key}_output"] = escaped_output

        if changed_files:
            context["files_to_stage"] = ", ".join(changed_files)

    steps: List[Tuple[Union[int, float], str, str]] = [
        (1, "duplicate", "Searching for duplicate issues"),
        (2, "docs", "Checking documentation for user error"),
        (3, "triage", "Assessing information completeness"),
        (4, "reproduce", "Attempting to reproduce the bug"),
        (5, "root_cause", "Analyzing root cause"),
        (5.5, "prompt_classification", "Classifying defect: code bug vs prompt defect"),
        (6, "test_plan", "Designing test strategy"),
        (7, "generate", "Generating failing unit test"),
        (8, "verify", "Verifying test catches the bug"),
        (9, "e2e_test", "Generating and running E2E tests"),
        (10, "pr", "Creating draft PR"),
    ]

    if last_completed_step == 5:
        start_step: Union[int, float] = 5.5
    elif last_completed_step == 5.5:
        start_step = 6
    else:
        start_step = last_completed_step + 1

    last_completed_step_to_save: Union[int, float] = last_completed_step
    consecutive_provider_failures = 0

    for step_num, name, description in steps:

        if step_num < start_step:
            continue

        if step_num == 5.5:
            if worktree_path is None:
                try:
                    branch_res = subprocess.run(
                        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                        cwd=cwd,
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    current_branch = branch_res.stdout.strip()
                    if current_branch not in ["main", "master"] and not quiet:
                        console.print(f"[yellow]Note: Creating branch from HEAD ({current_branch}), not origin/main. PR will include commits from this branch. Run from main for independent changes.[/yellow]")
                except subprocess.CalledProcessError:
                    pass

                wt_path, err = _setup_worktree(cwd, issue_number, quiet, resume_existing=False)
                if not wt_path:
                    return False, f"Failed to create worktree: {err}", total_cost, last_model_used, changed_files

                worktree_path = wt_path
                current_cwd = worktree_path
                context["worktree_path"] = str(worktree_path)

                if not quiet:
                    console.print(f"[blue]Working in worktree: {worktree_path}[/blue]")

        step_display = f"{step_num}" if isinstance(step_num, int) else f"{step_num}"
        step_suffix = str(step_num).replace(".", "_")

        if not quiet:
            console.print(f"[bold][Step {step_display}/11][/bold] {description}...")

        template_name = f"agentic_bug_step{step_suffix}_{name}_LLM"
        prompt_template = load_prompt_template(template_name)
        
        if not prompt_template:
            return False, f"Missing prompt template: {template_name}", total_cost, last_model_used, changed_files

        exclude_keys = list(context.keys())
        prompt_template = preprocess(prompt_template, recursive=True, double_curly_brackets=True, exclude_keys=exclude_keys)

        try:
            formatted_prompt = prompt_template.format(**context)
        except KeyError as e:
            msg = f"Prompt formatting error in step {step_num}: missing key {e}"
            if not quiet:
                console.print(f"[red]Error: {msg}[/red]")
            return False, msg, total_cost, last_model_used, changed_files

        success, output, cost, model = run_agentic_task(
            instruction=formatted_prompt,
            cwd=current_cwd,
            verbose=verbose,
            quiet=quiet,
            label=f"step{step_suffix}",
            timeout=BUG_STEP_TIMEOUTS.get(step_num, 340.0) + timeout_adder,
            max_retries=DEFAULT_MAX_RETRIES,
        )

        total_cost += cost
        last_model_used = model
        escaped_output = output.replace("{", "{{").replace("}", "}}").replace("\n", "\\n")
        context[f"step{step_suffix}_output"] = escaped_output

        if step_num == 1 and "Duplicate of #" in output:
            msg = f"Stopped at Step 1: Issue is a duplicate. {output.strip()}"
            if not quiet:
                console.print(f"⏹️  {msg}")
            return False, msg, total_cost, last_model_used, changed_files

        if step_num == 2:
            if "Feature Request (Not a Bug)" in output:
                msg = "Stopped at Step 2: Identified as Feature Request."
                if not quiet: console.print(f"⏹️  {msg}")
                return False, msg, total_cost, last_model_used, changed_files
            if "User Error (Not a Bug)" in output:
                msg = "Stopped at Step 2: Identified as User Error."
                if not quiet: console.print(f"⏹️  {msg}")
                return False, msg, total_cost, last_model_used, changed_files

        if step_num == 3 and "Needs More Info" in output:
            msg = "Stopped at Step 3: Insufficient information provided."
            if not quiet: console.print(f"⏹️  {msg}")
            return False, msg, total_cost, last_model_used, changed_files

        if step_num == 5.5 and "PROMPT_REVIEW:" in output:
            for line in output.splitlines():
                if line.startswith("PROMPT_REVIEW:"):
                    reason = line.split(":", 1)[1].strip()
                    break
            else:
                reason = "Prompt defect needs human review"
            msg = f"Stopped at Step 5.5: {reason}"
            if not quiet: console.print(f"⏹️  {msg}")
            return False, msg, total_cost, last_model_used, changed_files

        if step_num == 5.5:
            for line in output.splitlines():
                if line.startswith("PROMPT_FIXED:"):
                    fixed_file = line.split(":", 1)[1].strip()
                    if fixed_file and fixed_file not in changed_files:
                        changed_files.append(fixed_file)
                        context["files_to_stage"] = ", ".join(changed_files)
                    break

        if step_num == 7:
            extracted_files = []
            for line in output.splitlines():
                if line.startswith("FILES_CREATED:") or line.startswith("FILES_MODIFIED:"):
                    file_list = line.split(":", 1)[1].strip()
                    extracted_files.extend([f.strip() for f in file_list.split(",") if f.strip()])
            
            changed_files.extend(extracted_files)
            changed_files = list(dict.fromkeys(changed_files))
            context["files_to_stage"] = ", ".join(changed_files)

            if not changed_files:
                msg = "Stopped at Step 7: No test file generated."
                if not quiet: console.print(f"⏹️  {msg}")
                return False, msg, total_cost, last_model_used, changed_files

        if step_num == 8 and "FAIL: Test does not work as expected" in output:
            msg = "Stopped at Step 8: Generated test does not fail correctly (verification failed)."
            if not quiet: console.print(f"⏹️  {msg}")
            return False, msg, total_cost, last_model_used, changed_files

        if step_num == 9:
            if "E2E_FAIL: Test does not catch bug correctly" in output:
                msg = "Stopped at Step 9: E2E test does not catch bug correctly."
                if not quiet: console.print(f"⏹️  {msg}")
                return False, msg, total_cost, last_model_used, changed_files
            
            e2e_files = []
            for line in output.splitlines():
                if line.startswith("E2E_FILES_CREATED:"):
                    file_list = line.split(":", 1)[1].strip()
                    e2e_files.extend([f.strip() for f in file_list.split(",") if f.strip()])
            
            if e2e_files:
                changed_files.extend(e2e_files)
                context["files_to_stage"] = ", ".join(changed_files)

        if not success and not quiet:
            console.print(f"[yellow]Warning: Step {step_num} reported failure, but proceeding as no hard stop condition met.[/yellow]")
        elif not quiet:
            console.print(f"  → Step {step_num} complete.")

        if success:
            step_outputs[str(step_num)] = output
            last_completed_step_to_save = step_num
            consecutive_provider_failures = 0
        else:
            step_outputs[str(step_num)] = f"FAILED: {output}"
            if "All agent providers failed" in output:
                consecutive_provider_failures += 1
                if consecutive_provider_failures >= 3:
                    new_state = {
                        "workflow": "bug",
                        "issue_number": issue_number,
                        "issue_url": issue_url,
                        "last_completed_step": last_completed_step_to_save,
                        "step_outputs": step_outputs.copy(),
                        "total_cost": total_cost,
                        "model_used": last_model_used,
                        "changed_files": changed_files.copy(),
                        "worktree_path": str(worktree_path) if worktree_path else None,
                        "github_comment_id": github_comment_id
                    }
                    save_workflow_state(
                        cwd=cwd, issue_number=issue_number, workflow_type="bug",
                        state=new_state, state_dir=state_dir,
                        repo_owner=repo_owner, repo_name=repo_name,
                        use_github_state=use_github_state, github_comment_id=github_comment_id
                    )
                    return False, f"Aborting: {consecutive_provider_failures} consecutive steps failed — agent providers unavailable", total_cost, last_model_used, changed_files
            else:
                consecutive_provider_failures = 0

        new_state = {
            "workflow": "bug",
            "issue_number": issue_number,
            "issue_url": issue_url,
            "last_completed_step": last_completed_step_to_save,
            "step_outputs": step_outputs.copy(),
            "total_cost": total_cost,
            "model_used": last_model_used,
            "changed_files": changed_files.copy(),
            "worktree_path": str(worktree_path) if worktree_path else None,
            "github_comment_id": github_comment_id
        }
        
        # Retry logic for state saving
        for attempt in range(3):
            try:
                github_comment_id = save_workflow_state(
                    cwd=cwd,
                    issue_number=issue_number,
                    workflow_type="bug",
                    state=new_state,
                    state_dir=state_dir,
                    repo_owner=repo_owner,
                    repo_name=repo_name,
                    use_github_state=use_github_state,
                    github_comment_id=github_comment_id
                )
                break
            except Exception as e:
                if attempt == 2:
                    console.print(f"[red]Error: Failed to sync state to GitHub after 3 attempts: {e}[/red]")

    clear_workflow_state(
        cwd=cwd,
        issue_number=issue_number,
        workflow_type="bug",
        state_dir=state_dir,
        repo_owner=repo_owner,
        repo_name=repo_name,
        use_github_state=use_github_state
    )

    final_msg = "Investigation complete"
    if not quiet:
        console.print(f"✅ {final_msg}")
        console.print(f"   Total cost: ${total_cost:.4f}")
        console.print(f"   Files changed: {', '.join(changed_files)}")
        if worktree_path:
            console.print(f"   Worktree: {worktree_path}")

    return True, final_msg, total_cost, last_model_used, changed_files

if __name__ == "__main__":
    pass
