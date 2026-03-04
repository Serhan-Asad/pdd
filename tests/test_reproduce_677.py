"""
Reproduction test for GitHub issue #677:
pdd update generates inconsistent include paths across prompts.

The bug: when `pdd update` generates/updates prompts via LLM, the resulting
<include> paths are inconsistent across prompts — using 6 different styles
to reference the same file. Some paths reference non-existent files.

Root cause: Neither the update prompt template nor any post-processing step
enforces a canonical include path format or validates that referenced files exist.

This test demonstrates:
1. The LLM output is used as-is without include path normalization
2. There is no validation that <include> paths reference real files
3. The same preamble file can be referenced with wildly different path styles
"""

import os
import re
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from pydantic import BaseModel, Field

from pdd.update_prompt import update_prompt, PromptUpdate


# --- Scenario setup ---
# Simulates 6 LLM responses for different prompt files, each referencing the
# SAME preamble file but with inconsistent path styles (as reported in #677).

PREAMBLE_REAL_PATH = "prompts/frontend/context/preamble.prompt"

# These are the 6 different path styles the LLM generates for the same file
INCONSISTENT_INCLUDE_PATHS = [
    "prompts/frontend/context/preamble.prompt",       # absolute from project root
    "../../../../../context/preamble.prompt",          # deep relative
    "../../context/preamble.prompt",                   # shallow relative
    "context/project_preamble.prompt",                 # HALLUCINATED — file doesn't exist
    "./context/preamble.prompt",                       # dot-relative
    "../../../prompts/frontend/context/preamble.prompt",  # mixed relative+absolute
]


def _make_mock_prompt_with_include(include_path: str) -> str:
    """Build a mock updated prompt containing an <include> tag."""
    return (
        f"% Role: Frontend component\n"
        f"% Dependencies:\n"
        f"<preamble>\n"
        f"  <include>{include_path}</include>\n"
        f"</preamble>\n"
        f"% Requirements:\n"
        f"1. Render the page\n"
    )


class TestIssue677InconsistentIncludePaths:
    """Reproduce: pdd update generates inconsistent include paths."""

    def test_update_prompt_does_not_normalize_include_paths(self):
        """
        REPRODUCES BUG: update_prompt() returns LLM output as-is without
        normalizing <include> paths. Each call can produce a different path
        style for the same file.

        Expected behavior (after fix): all include paths should use a
        consistent convention.
        """
        collected_include_paths = []

        for i, include_path in enumerate(INCONSISTENT_INCLUDE_PATHS):
            mock_prompt_output = _make_mock_prompt_with_include(include_path)

            mock_first_response = {
                'result': f'Analysis of prompt {i}: needs preamble include',
                'cost': 0.001,
                'model_name': 'mock-model',
            }
            mock_second_response = {
                'result': PromptUpdate.model_construct(
                    modified_prompt=mock_prompt_output
                ),
                'cost': 0.002,
                'model_name': 'mock-model',
            }

            call_count = [0]
            def mock_llm_invoke(*args, **kwargs):
                call_count[0] += 1
                if 'output_pydantic' not in kwargs:
                    return mock_first_response
                return mock_second_response

            with patch('pdd.update_prompt.llm_invoke', side_effect=mock_llm_invoke), \
                 patch('pdd.update_prompt.load_prompt_template', return_value="template {input_prompt} {input_code} {modified_code}"), \
                 patch('pdd.update_prompt.preprocess', side_effect=lambda t, *a, **k: t):

                result_prompt, cost, model = update_prompt(
                    input_prompt="original prompt",
                    input_code="original code here",
                    modified_code="modified code here",
                    strength=0.7,
                    temperature=0.5,
                )

            # Extract the include path from the result
            match = re.search(r'<include>(.*?)</include>', result_prompt)
            assert match, f"No <include> tag found in output for iteration {i}"
            collected_include_paths.append(match.group(1))

        # BUG DEMONSTRATION: All 6 paths reference the same conceptual file
        # but use different styles. After a fix, they should all be identical.
        unique_paths = set(collected_include_paths)

        # This assertion FAILS — demonstrating the bug.
        # If the bug were fixed, all paths would be normalized to one canonical form.
        assert len(unique_paths) > 1, (
            "If this passes, the paths are still inconsistent (bug present). "
            f"Got {len(unique_paths)} unique path styles: {unique_paths}"
        )
        # The real assertion we WANT to pass after a fix:
        # assert len(unique_paths) == 1, f"Expected 1 canonical path, got {unique_paths}"

    def test_update_prompt_allows_hallucinated_paths(self):
        """
        REPRODUCES BUG: update_prompt() does not validate that <include> paths
        reference files that actually exist. The LLM can hallucinate paths like
        'context/project_preamble.prompt' (which doesn't exist).
        """
        hallucinated_path = "context/project_preamble.prompt"
        mock_prompt = _make_mock_prompt_with_include(hallucinated_path)

        mock_first_response = {
            'result': 'Analysis: needs preamble',
            'cost': 0.001,
            'model_name': 'mock-model',
        }
        mock_second_response = {
            'result': PromptUpdate.model_construct(modified_prompt=mock_prompt),
            'cost': 0.002,
            'model_name': 'mock-model',
        }

        def mock_llm_invoke(*args, **kwargs):
            if 'output_pydantic' not in kwargs:
                return mock_first_response
            return mock_second_response

        with patch('pdd.update_prompt.llm_invoke', side_effect=mock_llm_invoke), \
             patch('pdd.update_prompt.load_prompt_template', return_value="template {input_prompt} {input_code} {modified_code}"), \
             patch('pdd.update_prompt.preprocess', side_effect=lambda t, *a, **k: t):

            result_prompt, cost, model = update_prompt(
                input_prompt="original prompt",
                input_code="original code here",
                modified_code="modified code here",
                strength=0.7,
                temperature=0.5,
            )

        # The hallucinated path passes through without any validation
        assert hallucinated_path in result_prompt, (
            "Expected hallucinated path to be in output (no validation exists)"
        )

        # Verify the file doesn't actually exist (confirming it's hallucinated)
        project_root = Path(__file__).resolve().parent.parent
        assert not (project_root / hallucinated_path).exists(), (
            f"'{hallucinated_path}' unexpectedly exists — adjust test"
        )

        # BUG: This should have been caught. After fix, update_prompt should
        # either reject or correct include paths to non-existent files.

    def test_agentic_update_template_lacks_path_convention(self):
        """
        REPRODUCES BUG: The agentic_update_LLM.prompt template does not specify
        a canonical include path format. It says 'write them to context/ or
        equivalent directory' which is ambiguous.
        """
        from pdd.load_prompt_template import load_prompt_template

        template = load_prompt_template("agentic_update_LLM")
        assert template is not None, "Could not load agentic_update_LLM template"

        # Check that the template does NOT enforce a path convention
        # (this is the root cause of inconsistency)
        has_path_convention = any(phrase in template.lower() for phrase in [
            "paths must be relative to",
            "always use paths relative to",
            "canonical path",
            "path format:",
            "include paths should be relative to the prompt file",
            "include paths should be relative to the project root",
        ])

        # BUG: No path convention is specified
        assert not has_path_convention, (
            "Template now has path convention guidance — bug may be fixed"
        )

    def test_legacy_update_template_lacks_path_convention(self):
        """
        REPRODUCES BUG: The update_prompt_LLM.prompt template shows an example
        'context/orders_service_example.py' but doesn't enforce it as a rule.
        """
        from pdd.load_prompt_template import load_prompt_template

        template = load_prompt_template("update_prompt_LLM")
        assert template is not None, "Could not load update_prompt_LLM template"

        # The template shows examples but doesn't mandate a convention
        has_enforcement = any(phrase in template.lower() for phrase in [
            "must use",
            "always use paths relative to",
            "never use absolute",
            "path convention",
            "canonical include path",
        ])

        # BUG: No enforcement — only a soft example
        assert not has_enforcement, (
            "Template now enforces path convention — bug may be fixed"
        )

    def test_no_post_processing_of_include_paths_in_update_prompt(self):
        """
        REPRODUCES BUG: update_prompt() has no post-processing step to
        validate or normalize <include> paths in the LLM's output.

        The function returns second_response['result'].modified_prompt directly
        without any include path validation.
        """
        import inspect
        from pdd.update_prompt import update_prompt

        source = inspect.getsource(update_prompt)

        # Check that the function doesn't contain any include path normalization
        has_include_normalization = any(phrase in source for phrase in [
            'normalize_include',
            'validate_include',
            'fix_include',
            'resolve_include',
            'check_include',
            '<include>',
        ])

        # BUG: No include path handling exists in the function
        assert not has_include_normalization, (
            "update_prompt() now has include path handling — bug may be fixed"
        )
