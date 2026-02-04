import pytest
import os
from pathlib import Path
from pdd.preprocess import preprocess

# Define a temporary directory for test artifacts
@pytest.fixture
def temp_test_dir(tmp_path):
    """Fixture to provide a temporary directory for tests."""
    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(original_cwd)

def test_command_injection_vulnerability_present(temp_test_dir):
    """
    Verifies that the command injection vulnerability via <shell> tags is present.
    This test ensures that a malicious command embedded in a <shell> tag IS executed.
    
    On buggy code: This test would PASS because the malicious command would execute,
    creating 'malicious_file.txt'.
    On fixed code: This test would FAIL because the file would not be created.
    """
    malicious_file = temp_test_dir / "malicious_file.txt"
    malicious_command = f"echo EVIL > {malicious_file}"
    prompt = f"Before <shell>{malicious_command}</shell> After"

    result = preprocess(prompt, recursive=False, double_curly_brackets=False)

    # Assert that the malicious file WAS created
    assert malicious_file.exists()
    
    # Assert that the output contains the command's stdout (which is empty for this command)
    assert result == "Before  After"
