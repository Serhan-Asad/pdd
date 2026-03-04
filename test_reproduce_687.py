"""Reproduction test for issue #687: code_generator returns wrong model name after postprocessing."""

import unittest
from unittest.mock import patch, MagicMock


class TestCodeGeneratorModelNameBug(unittest.TestCase):
    """Reproduce: code_generator returns pre-postprocessing model name instead of post-processing model name."""

    @patch("pdd.code_generator.postprocess")
    @patch("pdd.code_generator.unfinished_prompt")
    @patch("pdd.code_generator.llm_invoke")
    @patch("pdd.code_generator.preprocess")
    def test_returns_postprocess_model_name(self, mock_preprocess, mock_llm_invoke, mock_unfinished, mock_postprocess):
        """When postprocessing uses a different model, the returned model name should reflect the postprocessor's model."""
        from pdd.code_generator import code_generator

        # Setup: preprocess returns the prompt unchanged
        mock_preprocess.return_value = "processed prompt"

        # Setup: initial LLM call returns a dict with model "gpt-4o"
        mock_llm_invoke.return_value = {
            "result": "```python\nprint('hello')\n```",
            "cost": 0.01,
            "model_name": "gpt-4o",
        }

        # Setup: unfinished_prompt says generation is complete (returns 4-tuple)
        mock_unfinished.return_value = ("Looks complete", True, 0.001, "gpt-4o-mini")

        # Setup: postprocess uses a DIFFERENT model "claude-3-5-sonnet"
        mock_postprocess.return_value = ("print('hello')", 0.005, "claude-3-5-sonnet")

        # Act
        runnable_code, total_cost, model_name = code_generator(
            prompt="Write hello world",
            language="python",
            strength=0.5,
            temperature=0.0,
            verbose=False,
        )

        # Assert: model_name should be from postprocessor, not initial generation
        print(f"\nInitial generation model: gpt-4o")
        print(f"Postprocessor model:      claude-3-5-sonnet")
        print(f"Returned model_name:      {model_name}")

        # This assertion will FAIL due to the bug - it returns "gpt-4o" instead of "claude-3-5-sonnet"
        self.assertEqual(
            model_name,
            "claude-3-5-sonnet",
            f"BUG REPRODUCED: Expected 'claude-3-5-sonnet' (postprocessor model) but got '{model_name}' (initial generation model)"
        )


if __name__ == "__main__":
    unittest.main()
