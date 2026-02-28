import json
import re
from typing import Tuple, Optional
from rich.console import Console
from . import EXTRACTION_STRENGTH
from .preprocess import preprocess
from .llm_invoke import llm_invoke
from .unfinished_prompt import unfinished_prompt
from .continue_generation import continue_generation
from .postprocess import postprocess

console = Console()

def code_generator(
    prompt: str,
    language: str,
    strength: float,
    temperature: float = 0.0,
    time: Optional[float] = None,
    verbose: bool = False,
    preprocess_prompt: bool = True,
    output_schema: Optional[dict] = None,
) -> Tuple[str, float, str]:
    """
    Generate code from a prompt using a language model.

    Args:
        prompt (str): The raw prompt to be processed
        language (str): The target programming language
        strength (float): The strength of the LLM model (0 to 1)
        temperature (float, optional): The temperature for the LLM model. Defaults to 0.0
        time (Optional[float], optional): The time for the LLM model. Defaults to None
        verbose (bool, optional): Whether to print detailed information. Defaults to False
        preprocess_prompt (bool, optional): Whether to preprocess the prompt. Defaults to True
        output_schema (Optional[dict], optional): JSON schema to enforce structured output. Defaults to None

    Returns:
        Tuple[str, float, str]: Tuple containing (runnable_code, total_cost, model_name)

    Raises:
        ValueError: If input parameters are invalid
        Exception: For other unexpected errors
    """
    try:
        # Input validation
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("Language must be a non-empty string")
        if not 0 <= strength <= 1:
            raise ValueError("Strength must be between 0 and 1")
        if not 0 <= temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")

        total_cost = 0.0
        model_name = ""

        # Step 1: Preprocess the prompt
        if preprocess_prompt:
            if verbose:
                console.print("[bold blue]Step 1: Preprocessing prompt[/bold blue]")

            processed_prompt = preprocess(prompt, recursive=False, double_curly_brackets=True)
        else:
            processed_prompt = prompt

        # Step 2: Generate initial response
        if verbose:
            console.print("[bold blue]Step 2: Generating initial response[/bold blue]")
        
        if 'data:image' in processed_prompt:
            parts = re.split(r'(data:image/[^;]+;base64,[A-Za-z0-9+/=]+)', processed_prompt)
            
            content = []
            for part in parts:
                if part.startswith('data:image'):
                    content.append({"type": "image_url", "image_url": {"url": part}})
                elif part != "":
                    content.append({"type": "text", "text": part})
            
            messages = [{"role": "user", "content": content}]

            response = llm_invoke(
                messages=messages,
                strength=strength,
                temperature=temperature,
                time=time,
                verbose=verbose,
                output_schema=output_schema,
                language=language,
            )
        else:
            response = llm_invoke(
                prompt=processed_prompt,
                input_json={},
                strength=strength,
                temperature=temperature,
                time=time,
                verbose=verbose,
                output_schema=output_schema,
                language=language,
            )
        initial_output = response['result']
        total_cost += response['cost']
        model_name = response['model_name']

        # Step 3: Check if generation is complete
        if verbose:
            console.print("[bold blue]Step 3: Checking completion status[/bold blue]")
        last_chunk = initial_output[-600:] if len(initial_output) > 600 else initial_output
        reasoning, is_finished, check_cost, _ = unfinished_prompt(
            prompt_text=last_chunk,
            strength=0.5,
            temperature=0.0,
            time=time,
            language=language,
            verbose=verbose
        )
        total_cost += check_cost

        # Step 3a: Continue generation if incomplete
        if not is_finished:
            if verbose:
                console.print("[bold yellow]Generation incomplete, continuing...[/bold yellow]")
            final_output, continue_cost, continue_model = continue_generation(
                formatted_input_prompt=processed_prompt,
                llm_output=initial_output,
                strength=strength,
                temperature=temperature,
                time=time,
                language=language,
                verbose=verbose
            )
            total_cost += continue_cost
            model_name = continue_model
        else:
            final_output = initial_output

        # Step 4: Postprocess the output
        if verbose:
            console.print("[bold blue]Step 4: Postprocessing output[/bold blue]")

        # For structured JSON targets, skip extract_code to avoid losing or altering schema-constrained payloads.
        if (isinstance(language, str) and language.strip().lower() == "json") or output_schema:
            if isinstance(final_output, str):
                runnable_code = final_output
            else:
                runnable_code = json.dumps(final_output)
            postprocess_cost = 0.0
            model_name_post = model_name
        else:
            runnable_code, postprocess_cost, model_name_post = postprocess(
                llm_output=final_output,
                language=language,
                strength=EXTRACTION_STRENGTH,
                temperature=0.0,
                time=time,
                verbose=verbose
            )
            total_cost += postprocess_cost

        # Issue #625: Validate function call argument counts for Python code
        if isinstance(language, str) and language.strip().lower() == 'python' and runnable_code:
            arg_mismatches = _validate_python_function_args_inline(runnable_code, verbose)
            if arg_mismatches:
                console.print(
                    "[bold yellow]Warning: Function argument mismatch detected in generated code. "
                    "Returning empty result to prevent writing invalid code.[/bold yellow]"
                )
                return "", total_cost, model_name

        return runnable_code, total_cost, model_name

    except ValueError as ve:
        if verbose:
            console.print(f"[bold red]Validation Error: {str(ve)}[/bold red]")
        raise
    except Exception as e:
        if verbose:
            console.print(f"[bold red]Unexpected Error: {str(e)}[/bold red]")
        raise


def _validate_python_function_args_inline(code: str, verbose: bool = False) -> list[dict]:
    """
    Validate that function calls in generated Python code match function definitions.

    This is an inline version of _validate_python_function_args from sync_orchestration
    that works on code strings rather than file paths. Used in the standalone
    code_generator path to catch mismatches before returning.

    Returns:
        List of dicts describing mismatches (empty if none found).
    """
    import ast

    if not code or not code.strip():
        return []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    # Collect function definitions, skipping class methods
    func_signatures: dict[str, dict] = {}

    def _collect_functions(nodes, inside_class=False):
        """Collect function definitions, skipping class methods."""
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not inside_class:
                    args = node.args
                    num_params = len(args.args)
                    num_defaults = len(args.defaults)
                    has_vararg = args.vararg is not None
                    has_kwonly = len(args.kwonlyargs) > 0

                    # If function uses keyword-only params, skip — positional
                    # count alone can't validate calls reliably.
                    if has_kwonly:
                        continue

                    min_params = num_params - num_defaults
                    max_params = None if has_vararg else num_params

                    func_signatures[node.name] = {
                        'min_params': min_params,
                        'max_params': max_params,
                    }
                _collect_functions(node.body, inside_class=False)
            elif isinstance(node, ast.ClassDef):
                _collect_functions(node.body, inside_class=True)

    _collect_functions(tree.body)

    if not func_signatures:
        return []

    mismatches: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue

        func_name = node.func.id
        if func_name not in func_signatures:
            continue

        sig = func_signatures[func_name]
        num_positional = len(node.args)
        num_keyword = len(node.keywords)
        total_args = num_positional + num_keyword

        # Skip validation when caller uses keyword args — we can't
        # reliably map keyword names to positional slots via AST alone.
        if num_keyword > 0:
            continue

        if total_args < sig['min_params']:
            mismatches.append({
                'function': func_name,
                'expected_min': sig['min_params'],
                'expected_max': sig['max_params'],
                'actual': total_args,
                'line': node.lineno,
            })
        elif sig['max_params'] is not None and total_args > sig['max_params']:
            mismatches.append({
                'function': func_name,
                'expected_min': sig['min_params'],
                'expected_max': sig['max_params'],
                'actual': total_args,
                'line': node.lineno,
            })

    if mismatches and verbose:
        for m in mismatches:
            expected_min = m['expected_min']
            expected_max = m['expected_max']
            if expected_max is None:
                expected_str = f"{expected_min}+"
            elif expected_min == expected_max:
                expected_str = str(expected_min)
            else:
                expected_str = f"{expected_min}-{expected_max}"
            console.print(
                f"[bold yellow]Warning: Function argument mismatch — {m['function']}() called with {m['actual']} args "
                f"but expects {expected_str} (line {m['line']})[/bold yellow]"
            )

    return mismatches
