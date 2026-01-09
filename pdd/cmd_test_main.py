import os
import click
import requests
from pathlib import Path
from rich import print as rprint
from rich.panel import Panel

from pdd.construct_paths import construct_paths
from pdd.generate_test import generate_test
from pdd.increase_tests import increase_tests
from pdd.core.cloud import CloudConfig
from pdd.config_resolution import resolve_effective_config

CLOUD_REQUEST_TIMEOUT = 400

def cmd_test_main(
    ctx: click.Context,
    prompt_file: str,
    code_file: str | None = None,
    output: str | None = None,
    example_file: str | None = None,
    language: str | None = None,
    coverage_report: str | None = None,
    existing_tests: list[str] | None = None,
    target_coverage: float | None = None,
    merge: bool = False,
    strength: float | None = None,
    temperature: float | None = None,
) -> tuple[str, float, str]:
    """
    CLI wrapper for generating or enhancing unit tests.
    """
    try:
        # 1. Validate inputs
        if (code_file and example_file) or (not code_file and not example_file):
            rprint("[bold red]Error: Exactly ONE of 'code_file' or 'example_file' must be provided (mutually exclusive).[/bold red]")
            return "", 0.0, "Error: code_file and example_file are mutually exclusive"

        if coverage_report and not existing_tests:
            rprint("[bold red]Error: 'existing_tests' is required when 'coverage_report' is provided.[/bold red]")
            return "", 0.0, "Error: Validation failed"

        is_tdd_mode = example_file is not None

        # 2. Build input paths for construct_paths
        # We map the CLI args to keys expected by construct_paths/input_strings
        input_file_paths = {"prompt_file": prompt_file}
        
        if is_tdd_mode:
            input_file_paths["example_file"] = example_file
        else:
            input_file_paths["code_file"] = code_file

        if coverage_report:
            input_file_paths["coverage_report"] = coverage_report
        
        # Note: We handle existing_tests reading manually below to ensure concatenation works as requested,
        # but we can pass the first one to construct_paths if we wanted path resolution. 
        # However, the prompt implies manual concatenation logic.

        command_options = {
            "output": output,
            "language": language,
            "merge": merge,
            "target_coverage": target_coverage
        }

        # 3. Call construct_paths
        resolved_config, input_strings, output_file_paths, detected_language = construct_paths(
            input_file_paths=input_file_paths,
            force=ctx.obj.get('force', False),
            quiet=ctx.obj.get('quiet', False),
            command="test",
            command_options=command_options,
            context_override=ctx.obj.get('context'),
            confirm_callback=ctx.obj.get('confirm_callback')
        )

        # 4. Resolve effective configuration
        # We pass the CLI overrides (strength, temperature) to resolve against defaults/config
        param_overrides = {}
        if strength is not None:
            param_overrides['strength'] = strength
        if temperature is not None:
            param_overrides['temperature'] = temperature
        
        # Time is retrieved from ctx.obj usually, but resolve_effective_config handles the hierarchy
        if ctx.obj.get('time') is not None:
            param_overrides['time'] = ctx.obj.get('time')

        final_config = resolve_effective_config(ctx, resolved_config, param_overrides=param_overrides)
        
        eff_strength = final_config['strength']
        eff_temperature = final_config['temperature']
        eff_time = final_config['time']
        verbose = ctx.obj.get('verbose', False)

        # 5. Prepare Content
        prompt_content = input_strings.get("prompt_file", "")
        
        if is_tdd_mode:
            code_content = None
            example_content = input_strings.get("example_file", "")
            source_path_str = example_file
        else:
            code_content = input_strings.get("code_file", "")
            example_content = None
            source_path_str = code_file

        # Handle existing tests concatenation
        existing_tests_content = None
        if existing_tests:
            concatenated = []
            for et_path in existing_tests:
                try:
                    p = Path(et_path).expanduser().resolve()
                    if p.exists():
                        concatenated.append(p.read_text(encoding="utf-8"))
                except Exception as e:
                    rprint(f"[yellow]Warning: Could not read existing test file {et_path}: {e}[/yellow]")
            if concatenated:
                existing_tests_content = "\n".join(concatenated)
                # Store in input_strings for consistency if needed later
                input_strings["existing_tests"] = existing_tests_content

        coverage_content = input_strings.get("coverage_report")

        # 6. Determine Metadata
        source_file_path = str(Path(source_path_str).expanduser().resolve())
        # Use the output path as-is from construct_paths
        test_file_path = output_file_paths['output']

        module_name = Path(source_file_path).stem
        if is_tdd_mode and module_name.endswith('_example'):
            module_name = module_name[:-8]  # Strip '_example'

        # 7. Execution Strategy (Cloud vs Local)
        generated_test_code = ""
        total_cost = 0.0
        model_name = "unknown"
        
        use_local = ctx.obj.get('local', False)
        cloud_success = False

        # --- Cloud Execution ---
        if not use_local:
            try:
                jwt_token = CloudConfig.get_jwt_token()
                if not jwt_token:
                    raise ValueError("Could not obtain JWT token")

                # Prepare Payload
                payload = {
                    "promptContent": prompt_content,
                    "language": detected_language,
                    "strength": eff_strength,
                    "temperature": eff_temperature,
                    "time": eff_time,
                    "verbose": verbose,
                    "sourceFilePath": source_file_path,
                    "testFilePath": test_file_path,
                    "moduleName": module_name,
                }

                if coverage_report:
                    payload["mode"] = "increase"
                    payload["codeContent"] = code_content # Required for increase
                    payload["existingTests"] = existing_tests_content
                    payload["coverageReport"] = coverage_content
                else:
                    payload["mode"] = "generate"
                    payload["codeContent"] = code_content
                    if is_tdd_mode:
                        payload["exampleContent"] = example_content

                if verbose:
                    rprint(Panel(f"Cloud Request: {payload.keys()}", title="Cloud Debug"))

                # Make Request
                response = requests.post(
                    f"{CloudConfig.API_BASE_URL}/generateTest",
                    json=payload,
                    headers={"Authorization": f"Bearer {jwt_token}"},
                    timeout=CLOUD_REQUEST_TIMEOUT
                )
                response.raise_for_status()
                data = response.json()

                generated_test_code = data.get("generatedTest", "")
                total_cost = float(data.get("totalCost", 0.0))
                model_name = data.get("modelName", "cloud-model")
                cloud_success = True

            except Exception as e:
                rprint(f"[yellow]Cloud execution failed: {e}. Falling back to local execution.[/yellow]")
                cloud_success = False

        # --- Local Execution Fallback ---
        if use_local or not cloud_success:
            if verbose:
                rprint(Panel("Executing locally...", title="Local Execution"))
            
            if coverage_report:
                # Augment existing tests
                generated_test_code, total_cost, model_name = increase_tests(
                    existing_unit_tests=existing_tests_content,
                    coverage_report=coverage_content,
                    code=code_content,
                    prompt_that_generated_code=prompt_content,
                    language=detected_language,
                    strength=eff_strength,
                    temperature=eff_temperature,
                    time=eff_time,
                    verbose=verbose
                )
            else:
                # Generate new tests
                generated_test_code, total_cost, model_name = generate_test(
                    prompt=prompt_content,
                    code=code_content,
                    example=example_content,
                    strength=eff_strength,
                    temperature=eff_temperature,
                    time=eff_time,
                    language=detected_language,
                    verbose=verbose,
                    source_file_path=source_file_path,
                    test_file_path=test_file_path,
                    module_name=module_name,
                    existing_tests=existing_tests_content
                )

        # 8. Post-process and Write Output
        if not generated_test_code or not generated_test_code.strip():
            rprint("[bold red]Error: Generated test content is empty.[/bold red]")
            return "", total_cost, f"Error: Empty output from {model_name}"

        # Determine output file and mode
        output_file = test_file_path
        mode = "w"
        content_to_write = generated_test_code

        if merge and existing_tests:
            # If merging, write to the first existing test file
            output_file = existing_tests[0]
            mode = "a"
            content_to_write = "\n\n" + generated_test_code

        # Ensure parent directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        try:
            with open(output_file, mode, encoding="utf-8") as f:
                f.write(content_to_write)

            if not ctx.obj.get('quiet'):
                action = "appended to" if mode == "a" else "saved to"
                rprint(f"[green]Unit tests {action}:[/green] {output_file}")
                
        except Exception as e:
            rprint(f"[bold red]Error writing output file: {e}[/bold red]")
            return generated_test_code, total_cost, f"Error: File I/O {e}"

        return generated_test_code, total_cost, model_name

    except click.Abort:
        raise
    except Exception as e:
        rprint(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        return "", 0.0, f"Error: {e}"