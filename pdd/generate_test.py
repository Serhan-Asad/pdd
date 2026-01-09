import os
import re
from typing import Optional, Tuple, Dict, Any
from rich.console import Console
from rich.markdown import Markdown

# Internal module imports based on the provided context
from pdd.load_prompt_template import load_prompt_template
from pdd.llm_invoke import llm_invoke
from pdd.preprocess import preprocess
from pdd.unfinished_prompt import unfinished_prompt
from pdd.continue_generation import continue_generation
from pdd.postprocess import postprocess

# Constants
DEFAULT_STRENGTH = 0.5
DEFAULT_TIME = 0.5
EXTRACTION_STRENGTH = 0.2  # Lower strength for extraction tasks to save cost/time
console = Console()

def _validate_inputs(
    prompt: str,
    code: str,
    strength: float,
    temperature: float,
    language: str
) -> None:
    """Validate input parameters."""
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Prompt must be a non-empty string")
    if not code or not isinstance(code, str):
        raise ValueError("Code must be a non-empty string")
    if not isinstance(strength, float) or not 0 <= strength <= 1:
        raise ValueError("Strength must be a float between 0 and 1")
    if not isinstance(temperature, float):
        raise ValueError("Temperature must be a float")
    if not language or not isinstance(language, str):
        raise ValueError("Language must be a non-empty string")

def generate_test(
    prompt: str,
    code: Optional[str] = None,
    strength: float = DEFAULT_STRENGTH,
    temperature: float = 0.0,
    time: float = DEFAULT_TIME,
    language: str = 'python',
    verbose: bool = False,
    source_file_path: Optional[str] = None,
    test_file_path: Optional[str] = None,
    module_name: Optional[str] = None,
    example: Optional[str] = None,
    existing_tests: Optional[str] = None
) -> Tuple[str, float, str]:
    """
    Generates a unit test from a code file or example using an LLM.

    Args:
        prompt (str): The prompt that generated the code or describes the requirement.
        code (Optional[str]): The code to generate a unit test from. Mutually exclusive with 'example'.
        strength (float): Strength of the LLM model (0-1).
        temperature (float): Temperature of the LLM model.
        time (float): Thinking effort for the LLM model (0-1).
        language (str): Language of the unit test.
        verbose (bool): Whether to print details.
        source_file_path (Optional[str]): Path to the code under test.
        test_file_path (Optional[str]): Destination path for the generated test.
        module_name (Optional[str]): Module name for imports.
        example (Optional[str]): Example content for TDD mode. Mutually exclusive with 'code'.
        existing_tests (Optional[str]): Content of existing tests to append to.

    Returns:
        Tuple[str, float, str]: (unit_test_code, total_cost, model_name)
    """
    total_cost = 0.0
    model_name = "unknown"

    # --- Step 1: Validation & Mode Determination ---
    # Validate mutual exclusivity
    has_code = code is not None and len(code.strip()) > 0
    has_example = example is not None and len(example.strip()) > 0

    if has_code and has_example:
        raise ValueError("Parameters 'code' and 'example' are mutually exclusive. Please provide only one.")
    if not has_code and not has_example:
        raise ValueError("Neither 'code' nor 'example' was provided")

    # Validate numeric ranges
    if not (0 <= strength <= 1):
        raise ValueError("Strength must be between 0 and 1.")
    if not (0 <= time <= 1):
        raise ValueError("Time must be between 0 and 1.")

    mode = "TDD (example-based)" if has_example else "Traditional (code-based)"

    # --- Step 2: Load Prompt Template ---
    template_name = "generate_test_LLM"
    template_content = load_prompt_template(template_name)
    if not template_content:
        raise ValueError(f"Failed to load {template_name} prompt template")

    # --- Step 3: Preprocessing ---
    # Preprocess prompt and template without recursion or double curly brackets
    # Note: We preprocess the input prompt to clean it. We preprocess the template 
    # to ensure it's clean, but we must ensure we don't break the {placeholders}.
    # The instruction says "without... doubling of the curly brackets", which implies
    # standard preprocessing that preserves single brackets for formatting.
    
    preprocessed_prompt = preprocess(prompt, recursive=False, double_curly_brackets=False)
    preprocessed_template = preprocess(template_content, recursive=False, double_curly_brackets=False)

    # --- Step 4: LLM Invocation ---
    input_variables = {
        "prompt_that_generated_code": preprocessed_prompt,
        "code": code if code else "",
        "example": example if example else "",
        "language": language,
        "source_file_path": source_file_path if source_file_path else "",
        "test_file_path": test_file_path if test_file_path else "",
        "module_name": module_name if module_name else "",
        "existing_tests": existing_tests if existing_tests else ""
    }

    if verbose:
        console.print(f"[bold blue]Generating test in {mode} mode...[/bold blue]")
        console.print(f"[dim]Strength: {strength}, Time: {time}, Temp: {temperature}[/dim]")

    try:
        llm_result = llm_invoke(
            prompt=preprocessed_template,
            input_json=input_variables,
            strength=strength,
            temperature=temperature,
            time=time,
            verbose=verbose
        )
    except Exception as e:
        raise RuntimeError(f"Failed to invoke LLM: {e}")

    current_text = llm_result.get('result', '')
    step_cost = llm_result.get('cost', 0.0)
    total_cost += step_cost
    model_name = llm_result.get('model_name', 'unknown')

    # --- Step 5: Verbose Output (Initial) ---
    if verbose:
        console.print("[bold green]Initial LLM Output:[/bold green]")
        console.print(Markdown(current_text))
        console.print(f"[dim]Initial Cost: ${step_cost:.6f}[/dim]")

    # --- Step 6: Unfinished Prompt Detection ---
    # Check the last 600 characters
    stripped_text = current_text.strip()
    
    if stripped_text:
        last_chunk = stripped_text[-600:]
        
        # Check if unfinished
        reasoning, is_finished, check_cost, _ = unfinished_prompt(
            prompt_text=last_chunk,
            strength=strength,
            temperature=temperature,
            time=time,
            language=language,
            verbose=verbose
        )
        total_cost += check_cost

        if not is_finished:
            if verbose:
                console.print(f"[yellow]Output detected as incomplete. Continuing generation...[/yellow]")
                console.print(f"[dim]Reasoning: {reasoning}[/dim]")

            # Prepare context for continuation
            # We need to reconstruct the full prompt context for the continuation
            # Usually continue_generation takes the original prompt and the current output
            # We format the template with inputs to get the full prompt sent to the LLM
            try:
                formatted_input_prompt = preprocessed_template.format(**input_variables)
            except KeyError as e:
                # Fallback if formatting fails due to template issues
                formatted_input_prompt = f"{preprocessed_template}\n\n[Inputs]: {input_variables}"

            cont_text, cont_cost, cont_model = continue_generation(
                formatted_input_prompt=formatted_input_prompt,
                llm_output=current_text,
                strength=strength,
                temperature=temperature,
                verbose=verbose
            )
            
            current_text = cont_text
            total_cost += cont_cost
            model_name = cont_model # Update model name to the one used for continuation

    # --- Step 7: Postprocessing ---
    if verbose:
        console.print("[bold blue]Post-processing result...[/bold blue]")

    extracted_code, pp_cost, _ = postprocess(
        llm_output=current_text,
        language=language,
        strength=EXTRACTION_STRENGTH,
        temperature=temperature,
        time=time,
        verbose=verbose
    )
    total_cost += pp_cost

    # Fallback extraction if postprocess returns empty or raw text that looks like markdown
    # (Simple heuristic: if result still has ``` and postprocess failed to clean it)
    if not extracted_code.strip() or "```" in extracted_code:
        # Try to find the largest code block
        code_blocks = re.findall(r'```(?:' + re.escape(language) + r')?(.*?)```', current_text, re.DOTALL)
        if code_blocks:
            # Pick the longest block as it's likely the test code
            extracted_code = max(code_blocks, key=len).strip()
        else:
            # If no blocks found, check if the raw text looks like code (has imports or defs)
            if "import " in current_text or "def test_" in current_text:
                # Use raw text but strip leading/trailing markdown if present
                extracted_code = current_text.replace("```" + language, "").replace("```", "").strip()

    # --- Step 8: Final Verbose ---
    if verbose:
        console.print(f"[bold green]Generation Complete.[/bold green]")
        console.print(f"[bold]Total Cost:[/bold] ${total_cost:.6f}")
        console.print(f"[bold]Model Used:[/bold] {model_name}")

    # --- Step 9: Return ---
    return extracted_code, total_cost, model_name