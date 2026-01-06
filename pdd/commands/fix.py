from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import click
from rich.console import Console

from ..fix_main import fix_main
from ..track_cost import track_cost
from ..core.errors import handle_error

console = Console()


@click.command(name="fix")
@click.argument("prompt_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("code_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output-test",
    type=click.Path(),
    help="Specify where to save the fixed unit test file.",
)
@click.option(
    "--output-code",
    type=click.Path(),
    help="Specify where to save the fixed code file.",
)
@click.option(
    "--output-results",
    type=click.Path(),
    help="Specify where to save the results of the error fixing process.",
)
@click.option(
    "--loop",
    is_flag=True,
    help="Enable iterative fixing process.",
)
@click.option(
    "--verification-program",
    type=click.Path(exists=True, dir_okay=False),
    help="Specify the path to a Python program that verifies if the code still runs correctly.",
)
@click.option(
    "--max-attempts",
    type=int,
    default=3,
    help="Set the maximum number of fix attempts before giving up.",
)
@click.option(
    "--budget",
    type=float,
    default=5.0,
    help="Set the maximum cost allowed for the fixing process.",
)
@click.option(
    "--auto-submit",
    is_flag=True,
    help="Automatically submit the example if all unit tests pass during the fix loop.",
)
@click.option(
    "--agentic-fallback/--no-agentic-fallback",
    default=True,
    help="Enable or disable the agentic fallback mode.",
)
@click.pass_context
@track_cost
def fix(
    ctx: click.Context,
    prompt_file: str,
    code_file: str,
    files: Tuple[str, ...],
    output_test: Optional[str],
    output_code: Optional[str],
    output_results: Optional[str],
    loop: bool,
    verification_program: Optional[str],
    max_attempts: int,
    budget: float,
    auto_submit: bool,
    agentic_fallback: bool,
) -> Tuple[Dict[str, Any], float, str]:
    """
    Fix errors in code and unit tests based on error messages and the original prompt.

    This command supports two modes:
    1. Non-Loop Mode (default): Requires an ERROR_FILE containing pre-captured errors.
       Usage: pdd fix PROMPT_FILE CODE_FILE UNIT_TEST_FILE... ERROR_FILE

    2. Loop Mode (--loop): Iteratively runs a verification program to capture errors and fix code.
       Usage: pdd fix --loop --verification-program PATH PROMPT_FILE CODE_FILE UNIT_TEST_FILE...
    """
    unit_test_files: List[str] = []
    error_file: str = ""

    # Determine mode and parse variable arguments
    if loop:
        if not verification_program:
            raise click.UsageError(
                "The --verification-program option is required when using --loop."
            )
        if not files:
            raise click.UsageError("At least one UNIT_TEST_FILE is required.")
        
        # In loop mode, all trailing files are test files
        unit_test_files = list(files)
        error_file = ""  # Not used in loop mode
    else:
        # Non-loop mode
        if not files:
            raise click.UsageError("UNIT_TEST_FILE(s) and ERROR_FILE are required.")
        if len(files) < 2:
            raise click.UsageError(
                "At least one UNIT_TEST_FILE and one ERROR_FILE are required."
            )

        # In non-loop mode, the last file is the error file
        unit_test_files = list(files[:-1])
        error_file = files[-1]

    # Initialize tracking variables
    total_cost = 0.0
    last_model = "unknown"
    results: Dict[str, Any] = {
        "success": True,
        "fixed_code": "",
        "fixed_test": "",
        "attempts": 0,
        "cost": 0.0,
        "model": "unknown",
    }

    # Process each test file
    for test_file in unit_test_files:
        remaining_budget = budget - total_cost
        
        if remaining_budget <= 0:
            console.print(
                "[bold red]Budget exhausted before processing all files.[/bold red]"
            )
            results["success"] = False
            break

        try:
            (
                success,
                f_test,
                f_code,
                attempts,
                cost,
                model,
            ) = fix_main(
                ctx=ctx,
                prompt_file=prompt_file,
                code_file=code_file,
                unit_test_file=test_file,
                error_file=error_file,
                output_test=output_test,
                output_code=output_code,
                output_results=output_results,
                loop=loop,
                verification_program=verification_program,
                max_attempts=max_attempts,
                budget=remaining_budget,
                auto_submit=auto_submit,
                agentic_fallback=agentic_fallback,
            )

            # Update cumulative stats
            total_cost += cost
            results["attempts"] += attempts
            results["cost"] = total_cost
            last_model = model

            # Update results with the latest fix content
            if success:
                results["fixed_code"] = f_code
                results["fixed_test"] = f_test
            else:
                results["success"] = False
                # We continue processing other files even if one fails, 
                # but mark the overall operation as failed.

        except Exception as e:
            handle_error(e, "fix", ctx.obj.get("quiet", False))
            results["success"] = False
            # If an exception occurs (e.g. file error), we stop processing
            break

    results["model"] = last_model

    # Return tuple required by @track_cost: (result_object, cost, model_name)
    return results, total_cost, last_model