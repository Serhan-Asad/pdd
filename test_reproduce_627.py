"""
Reproduction test for Issue #627:
pdd generate leaves hardcoded placeholder strings in generated code.

This test demonstrates that:
1. Code with placeholder config strings passes unfinished_prompt() as "finished"
2. The code_generator pipeline has no step to detect/flag placeholder values
3. Plausible-but-wrong config strings like "pdd-cloud-project-id" slip through
"""
import ast
import textwrap
from unittest.mock import patch, MagicMock

import pytest

from pdd.unfinished_prompt import unfinished_prompt


# --- Sample code that contains placeholder config strings ---
CODE_WITH_PLACEHOLDER = textwrap.dedent('''\
    import os
    from google.cloud import pubsub_v1

    PROJECT_ID = "pdd-cloud-project-id"

    def send_notification(topic: str, message: str) -> None:
        """Send a notification via Pub/Sub."""
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(PROJECT_ID, topic)
        future = publisher.publish(topic_path, message.encode("utf-8"))
        future.result()
''')

# --- Same code but using os.getenv (the correct pattern) ---
CODE_WITH_GETENV = textwrap.dedent('''\
    import os
    from google.cloud import pubsub_v1

    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

    def send_notification(topic: str, message: str) -> None:
        """Send a notification via Pub/Sub."""
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(PROJECT_ID, topic)
        future = publisher.publish(topic_path, message.encode("utf-8"))
        future.result()
''')

# More placeholder patterns that should be caught
ADDITIONAL_PLACEHOLDER_EXAMPLES = [
    'API_KEY = "your-api-key-here"',
    'DATABASE_URL = "postgres://user:password@localhost/dbname"',
    'SECRET_KEY = "change-me-in-production"',
    'PROJECT_ID = "my-project-id"',
    'BUCKET_NAME = "your-bucket-name"',
    'REGION = "us-central1"',  # This one is borderline - could be legit
]


class TestReproduceIssue627:
    """Reproduce the bug: placeholder strings pass through undetected."""

    def test_placeholder_code_passes_syntax_check(self):
        """
        REPRODUCES BUG: Code with hardcoded placeholder 'pdd-cloud-project-id'
        passes ast.parse and is marked as finished by unfinished_prompt().

        The unfinished_prompt() function only checks syntactic completeness,
        NOT whether config values are placeholders vs real values.
        """
        # The code with a placeholder is syntactically valid Python
        ast.parse(CODE_WITH_PLACEHOLDER)  # Should not raise

        # unfinished_prompt() marks it as finished (no LLM call needed)
        reasoning, is_finished, cost, model = unfinished_prompt(
            prompt_text=CODE_WITH_PLACEHOLDER,
            language="python",
        )

        # BUG: The code is marked as "finished" despite containing a placeholder
        assert is_finished is True
        assert model == "syntactic_check"
        assert cost == 0.0
        print(f"\nBUG REPRODUCED: Code with placeholder 'pdd-cloud-project-id' "
              f"marked as finished by {model}")
        print(f"Reasoning: {reasoning}")

    def test_correct_code_also_passes(self):
        """The correct code (with os.getenv) also passes - for comparison."""
        reasoning, is_finished, cost, model = unfinished_prompt(
            prompt_text=CODE_WITH_GETENV,
            language="python",
        )
        assert is_finished is True
        assert model == "syntactic_check"
        print(f"\nCorrect code (os.getenv) also passes: {model}")

    def test_no_placeholder_detection_in_pipeline(self):
        """
        REPRODUCES BUG: The full code_generator pipeline has no step that
        checks for placeholder config strings.

        We mock the LLM to return code with a placeholder, and verify
        it passes through the entire pipeline without being flagged.
        """
        from pdd.code_generator import code_generator

        mock_llm_response = {
            'result': CODE_WITH_PLACEHOLDER,
            'cost': 0.01,
            'model_name': 'mock-model',
        }

        with patch('pdd.code_generator.llm_invoke', return_value=mock_llm_response), \
             patch('pdd.code_generator.postprocess', return_value=(CODE_WITH_PLACEHOLDER, 0.0, 'mock-model')):

            result_code, cost, model = code_generator(
                prompt="Generate a notification service using Google Cloud Pub/Sub",
                language="python",
                strength=0.5,
                preprocess_prompt=False,
            )

            # BUG: The placeholder string survives the entire pipeline
            assert 'pdd-cloud-project-id' in result_code
            # BUG: No warning, no TODO marker, no detection at all
            assert 'TODO' not in result_code
            assert 'FIXME' not in result_code
            print(f"\nBUG REPRODUCED: Placeholder 'pdd-cloud-project-id' survived "
                  f"the entire code_generator pipeline undetected")
            print(f"Generated code contains: PROJECT_ID = \"pdd-cloud-project-id\"")

    def test_unfinished_prompt_has_no_placeholder_detection(self):
        """
        Verify that unfinished_prompt() has NO mechanism to detect
        placeholder config strings — it only checks syntax.
        """
        for placeholder_code in ADDITIONAL_PLACEHOLDER_EXAMPLES:
            full_code = f"import os\n\n{placeholder_code}\n\ndef main():\n    print('hello')\n"
            reasoning, is_finished, cost, model = unfinished_prompt(
                prompt_text=full_code,
                language="python",
            )
            # All placeholder patterns pass through undetected
            assert is_finished is True, (
                f"Expected placeholder to pass through undetected: {placeholder_code}"
            )

        print(f"\nBUG REPRODUCED: All {len(ADDITIONAL_PLACEHOLDER_EXAMPLES)} placeholder "
              f"patterns passed unfinished_prompt() undetected")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
