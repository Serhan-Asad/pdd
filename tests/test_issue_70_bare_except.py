"""
Tests for Issue #70: Bare except clauses can hide critical exceptions

This test file addresses the bug where 9 instances of bare `except:` clauses
throughout the codebase catch ALL exceptions including KeyboardInterrupt and
SystemExit, preventing proper program termination.

The primary test uses static AST analysis to detect bare except clauses,
providing regression prevention for this anti-pattern.
"""

import ast
import inspect
from pathlib import Path

import pytest


class TestStaticAnalysisBareExcept:
    """Static analysis to detect bare except clauses in the codebase."""

    def test_no_bare_except_clauses_in_codebase(self):
        """
        Verify that no bare except: clauses exist in the codebase.

        This test scans all Python files in the pdd/ directory using AST parsing
        to detect bare except clauses (ExceptHandler nodes with type=None).

        Expected: Zero bare except clauses found (all should use specific exception types)
        Actual (before fix): 9 bare except clauses at:
          - pdd/fix_errors_from_unit_tests.py:88, 95
          - pdd/construct_paths.py:266
          - pdd/update_main.py:65
          - pdd/unfinished_prompt.py:108
          - pdd/setup_tool.py:293
          - pdd/agentic_common.py:549
          - pdd/pin_example_hack.py:1659
          - pdd/sync_orchestration.py:1823

        This test FAILS on the current buggy code and will PASS once the fix is applied.
        """
        # Get the project root (parent of tests/ directory)
        project_root = Path(__file__).parent.parent
        pdd_dir = project_root / "pdd"

        bare_except_violations = []

        # Walk through all Python files in pdd/ directory
        for py_file in pdd_dir.rglob("*.py"):
            if py_file.is_file():
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source = f.read()

                    # Parse the file into an AST
                    tree = ast.parse(source, filename=str(py_file))

                    # Walk the AST looking for bare except clauses
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ExceptHandler):
                            # ExceptHandler.type is None for bare except:
                            if node.type is None:
                                # Calculate line number
                                line_num = node.lineno
                                relative_path = py_file.relative_to(project_root)
                                bare_except_violations.append(
                                    f"{relative_path}:{line_num}"
                                )
                except SyntaxError:
                    # Skip files with syntax errors (shouldn't happen in production code)
                    continue

        # Assert no bare except clauses found
        if bare_except_violations:
            violation_list = "\n  - ".join(bare_except_violations)
            pytest.fail(
                f"Found {len(bare_except_violations)} bare except: clause(s):\n"
                f"  - {violation_list}\n\n"
                f"Bare except: clauses catch ALL exceptions including KeyboardInterrupt "
                f"and SystemExit. Use specific exception types instead:\n"
                f"  - For file operations: except (OSError, FileNotFoundError):\n"
                f"  - For imports: except (ImportError, ModuleNotFoundError):\n"
                f"  - For JSON: except (json.JSONDecodeError, ValueError):\n"
                f"  - For general cleanup: except Exception:\n"
            )


class TestFileCleanupExceptionHandling:
    """Verify that file cleanup code will use specific exception types after fix."""

    def test_file_cleanup_has_specific_exception_types(self):
        """
        Document the expected fix for file cleanup code.

        The write_to_error_file() function has 2 bare except: clauses in cleanup code
        (lines 88 and 95). These should be replaced with specific exception types.

        This test will PASS once bare except: is fixed to except (OSError, FileNotFoundError):
        """
        from pdd.fix_errors_from_unit_tests import write_to_error_file

        # Get the source code of the function
        source = inspect.getsource(write_to_error_file)

        # Count bare except: clauses (this is a simple string check, not AST-based)
        # Before fix: Should find "except:" in the cleanup code
        # After fix: Should find "except (OSError, FileNotFoundError):" instead
        bare_except_count = source.count('except:\n')

        # Document the issue
        if bare_except_count > 0:
            pytest.fail(
                f"Found {bare_except_count} bare except: clause(s) in write_to_error_file(). "
                f"These should be replaced with specific exception types:\n"
                f"  except (OSError, FileNotFoundError):\n"
                f"This allows KeyboardInterrupt and SystemExit to propagate."
            )


# Summary comment for test execution
"""
Running these tests:

# Run all bare except tests
pytest tests/test_issue_70_bare_except.py -v

# Run only static analysis (primary regression test)
pytest tests/test_issue_70_bare_except.py::TestStaticAnalysisBareExcept -v

Expected results:
- Before fix: Static analysis test FAILS (finds 9 bare except clauses)
- Before fix: File cleanup test FAILS (finds 2 bare except clauses in write_to_error_file)
- After fix: All tests PASS

The static analysis test is the primary regression prevention mechanism.
It will fail CI builds if any new bare except: clauses are introduced.
"""
