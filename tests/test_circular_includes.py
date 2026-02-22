"""
Tests for circular <include> detection in pdd/preprocess.py (Issue #521).

The preprocessor has no cycle detection — circular includes recurse ~82 times
until Python's recursion limit, then the broad `except Exception` swallows
the RecursionError and returns corrupted output with exit code 0.

These tests verify that circular includes produce an error (exception or
error marker in output), NOT silently corrupted content.
"""

import os
import pytest
from unittest.mock import patch, mock_open, MagicMock
from pdd.preprocess import preprocess

# Store original so we can restore
_original_pdd_path = os.environ.get('PDD_PATH')


def set_pdd_path(path: str) -> None:
    os.environ['PDD_PATH'] = path


def _make_mock_open(file_map: dict):
    """Create a mock open that returns content based on filename."""
    def side_effect(file_name, *args, **kwargs):
        mock_file = MagicMock()
        for key, content in file_map.items():
            if key in str(file_name):
                mock_file.read.return_value = content
                mock_file.__enter__ = lambda s: s
                mock_file.__exit__ = MagicMock(return_value=False)
                return mock_file
        raise FileNotFoundError(f"No mock for {file_name}")
    return side_effect


class TestCircularIncludes:
    """Issue #521: Circular <include> tags must be detected, not silently expanded."""

    def setup_method(self):
        set_pdd_path('/mock/path')

    def teardown_method(self):
        if _original_pdd_path is not None:
            os.environ['PDD_PATH'] = _original_pdd_path
        elif 'PDD_PATH' in os.environ:
            del os.environ['PDD_PATH']

    def test_circular_xml_includes_must_error(self):
        """A→B→A circular include must raise or return error, not silently corrupt."""
        file_map = {
            'a.txt': 'Hello\n<include>b.txt</include>',
            'b.txt': 'World\n<include>a.txt</include>',
        }
        with patch('builtins.open', mock_open()) as m:
            m.side_effect = _make_mock_open(file_map)

            # The bug: preprocess silently returns corrupted output with
            # "Hello" repeated ~82 times. A correct implementation must
            # either raise an error OR return output containing an error marker.
            try:
                result = preprocess(
                    '<include>a.txt</include>',
                    recursive=True,
                    double_curly_brackets=False,
                )
            except (RecursionError, ValueError, RuntimeError):
                # Any of these exceptions is acceptable — cycle was detected
                return

            # If no exception, the output must NOT contain duplicated content.
            # The buggy behavior produces "Hello" 82+ times.
            hello_count = result.count('Hello')
            world_count = result.count('World')
            assert hello_count <= 2 and world_count <= 2, (
                f"Circular include silently produced corrupted output: "
                f"'Hello' appeared {hello_count} times, 'World' appeared {world_count} times. "
                f"Expected an error or at most 2 occurrences each."
            )

    def test_circular_backtick_includes_must_error(self):
        """Circular backtick-style includes must also be detected."""
        file_map = {
            'x.txt': 'Foo\n```<y.txt>```',
            'y.txt': 'Bar\n```<x.txt>```',
        }
        with patch('builtins.open', mock_open()) as m:
            m.side_effect = _make_mock_open(file_map)

            try:
                result = preprocess(
                    '```<x.txt>```',
                    recursive=True,
                    double_curly_brackets=False,
                )
            except (RecursionError, ValueError, RuntimeError):
                return

            foo_count = result.count('Foo')
            bar_count = result.count('Bar')
            assert foo_count <= 2 and bar_count <= 2, (
                f"Circular backtick include silently produced corrupted output: "
                f"'Foo' appeared {foo_count} times, 'Bar' appeared {bar_count} times."
            )

    def test_self_referencing_include_must_error(self):
        """A file that includes itself must be detected as circular."""
        file_map = {
            'self.txt': 'Content\n<include>self.txt</include>',
        }
        with patch('builtins.open', mock_open()) as m:
            m.side_effect = _make_mock_open(file_map)

            try:
                result = preprocess(
                    '<include>self.txt</include>',
                    recursive=True,
                    double_curly_brackets=False,
                )
            except (RecursionError, ValueError, RuntimeError):
                return

            content_count = result.count('Content')
            assert content_count <= 2, (
                f"Self-referencing include silently produced corrupted output: "
                f"'Content' appeared {content_count} times."
            )

    def test_three_file_cycle_must_error(self):
        """A→B→C→A three-file cycle must be detected."""
        file_map = {
            'a.txt': 'AAA\n<include>b.txt</include>',
            'b.txt': 'BBB\n<include>c.txt</include>',
            'c.txt': 'CCC\n<include>a.txt</include>',
        }
        with patch('builtins.open', mock_open()) as m:
            m.side_effect = _make_mock_open(file_map)

            try:
                result = preprocess(
                    '<include>a.txt</include>',
                    recursive=True,
                    double_curly_brackets=False,
                )
            except (RecursionError, ValueError, RuntimeError):
                return

            aaa_count = result.count('AAA')
            assert aaa_count <= 2, (
                f"Three-file circular include silently produced corrupted output: "
                f"'AAA' appeared {aaa_count} times."
            )

    def test_non_circular_recursive_includes_still_work(self):
        """Non-circular recursive includes (A→B→C, no cycle) must still work."""
        file_map = {
            'top.txt': 'Top\n<include>mid.txt</include>',
            'mid.txt': 'Mid\n<include>leaf.txt</include>',
            'leaf.txt': 'Leaf',
        }
        with patch('builtins.open', mock_open()) as m:
            m.side_effect = _make_mock_open(file_map)

            result = preprocess(
                '<include>top.txt</include>',
                recursive=True,
                double_curly_brackets=False,
            )

            assert 'Top' in result
            assert 'Mid' in result
            assert 'Leaf' in result

    def test_diamond_includes_not_falsely_flagged(self):
        """Diamond pattern (A→B, A→C, B→D, C→D) is NOT circular and must work."""
        file_map = {
            'a.txt': '<include>b.txt</include>\n<include>c.txt</include>',
            'b.txt': 'B\n<include>d.txt</include>',
            'c.txt': 'C\n<include>d.txt</include>',
            'd.txt': 'Shared',
        }
        with patch('builtins.open', mock_open()) as m:
            m.side_effect = _make_mock_open(file_map)

            result = preprocess(
                '<include>a.txt</include>',
                recursive=True,
                double_curly_brackets=False,
            )

            assert 'B' in result
            assert 'C' in result
            # D is included twice (via B and via C) — that's fine, not circular
            assert result.count('Shared') == 2


# ---------------------------------------------------------------------------
# Issue #157: Non-recursive (default) path has no cycle detection.
#
# All tests above use recursive=True. The non-recursive path iterates via a
# `while prev_text != current_text` loop that never converges on circular
# includes because `_seen` is never populated when recursive=False.
# These tests verify the non-recursive path also detects cycles.
#
# Design note: The bug causes an infinite loop (not a crash). We can't use
# signal.SIGALRM because the broad `except Exception` in replace_include()
# catches it and converts it to an error marker string. Instead, we use a
# counting mock that raises a BaseException subclass (which `except Exception`
# cannot catch) after too many open() calls — proving the loop ran away.
# ---------------------------------------------------------------------------

from pdd.preprocess import process_include_tags, process_backtick_includes


class _InfiniteLoopDetected(BaseException):
    """Raised by counting mock when open() is called too many times.

    Inherits from BaseException (not Exception) so the broad
    `except Exception` handler in replace_include() cannot catch it.
    """


def _make_counting_mock_open(file_map: dict, max_calls: int = 20):
    """Create a mock open that counts calls and raises after max_calls.

    For a 2-file circular include (A→B→A), the non-recursive while loop
    calls open() once per iteration. A linear chain A→B→C needs at most
    3 calls. So max_calls=20 is generous for legitimate use cases but
    catches runaway loops quickly.
    """
    call_count = [0]  # mutable counter

    def side_effect(file_name, *args, **kwargs):
        call_count[0] += 1
        if call_count[0] > max_calls:
            raise _InfiniteLoopDetected(
                f"open() called {call_count[0]} times — infinite loop detected. "
                f"Bug #157: _seen is never populated in non-recursive mode."
            )
        mock_file = MagicMock()
        for key, content in file_map.items():
            if key in str(file_name):
                mock_file.read.return_value = content
                mock_file.__enter__ = lambda s: s
                mock_file.__exit__ = MagicMock(return_value=False)
                return mock_file
        raise FileNotFoundError(f"No mock for {file_name}")

    return side_effect


class TestCircularIncludesNonRecursive:
    """Issue #157: Circular includes must be detected in non-recursive (default) mode."""

    def setup_method(self):
        set_pdd_path('/mock/path')

    def teardown_method(self):
        if _original_pdd_path is not None:
            os.environ['PDD_PATH'] = _original_pdd_path
        elif 'PDD_PATH' in os.environ:
            del os.environ['PDD_PATH']

    # -- XML <include> tag tests (process_include_tags, non-recursive) ------

    def test_circular_xml_ab_nonrecursive(self):
        """A→B→A circular XML include in non-recursive mode must raise ValueError, not hang."""
        file_map = {
            'a.txt': '<include>b.txt</include>',
            'b.txt': '<include>a.txt</include>',
        }
        with patch('builtins.open', mock_open()) as m:
            m.side_effect = _make_counting_mock_open(file_map)

            try:
                process_include_tags(
                    '<include>a.txt</include>',
                    recursive=False,
                )
            except ValueError as exc:
                # Correct behavior — cycle detected
                assert "circular" in str(exc).lower(), (
                    f"ValueError raised but not about circular includes: {exc}"
                )
                return
            except _InfiniteLoopDetected:
                pytest.fail(
                    "process_include_tags() looped 20+ times on circular A→B→A includes "
                    "in non-recursive mode. Bug #157: _seen is never populated when "
                    "recursive=False, so the while loop never converges."
                )

            # If function returned without error, check for silent corruption
            pytest.fail(
                "process_include_tags() returned without raising ValueError on circular "
                "A→B→A includes. Expected ValueError('Circular include detected: ...')."
            )

    def test_self_referencing_xml_nonrecursive(self):
        """A file that includes itself must be detected in non-recursive mode."""
        file_map = {
            'self.txt': 'Content\n<include>self.txt</include>',
        }
        with patch('builtins.open', mock_open()) as m:
            m.side_effect = _make_counting_mock_open(file_map)

            try:
                process_include_tags(
                    '<include>self.txt</include>',
                    recursive=False,
                )
            except ValueError as exc:
                assert "circular" in str(exc).lower()
                return
            except _InfiniteLoopDetected:
                pytest.fail(
                    "process_include_tags() looped 20+ times on self-referencing include "
                    "in non-recursive mode. Bug #157: no cycle detection when recursive=False."
                )

            pytest.fail(
                "process_include_tags() returned without raising ValueError on "
                "self-referencing include."
            )

    def test_three_file_cycle_xml_nonrecursive(self):
        """A→B→C→A three-file cycle must be detected in non-recursive mode."""
        file_map = {
            'a.txt': 'AAA\n<include>b.txt</include>',
            'b.txt': 'BBB\n<include>c.txt</include>',
            'c.txt': 'CCC\n<include>a.txt</include>',
        }
        with patch('builtins.open', mock_open()) as m:
            m.side_effect = _make_counting_mock_open(file_map)

            try:
                process_include_tags(
                    '<include>a.txt</include>',
                    recursive=False,
                )
            except ValueError as exc:
                assert "circular" in str(exc).lower()
                return
            except _InfiniteLoopDetected:
                pytest.fail(
                    "process_include_tags() looped 20+ times on A→B→C→A cycle in "
                    "non-recursive mode. Bug #157: _seen is never populated when "
                    "recursive=False."
                )

            pytest.fail(
                "process_include_tags() returned without raising ValueError on "
                "A→B→C→A circular include cycle."
            )

    # -- Backtick include tests (process_backtick_includes, non-recursive) --

    def test_circular_backtick_ab_nonrecursive(self):
        """A→B→A circular backtick include in non-recursive mode must raise, not hang."""
        file_map = {
            'x.txt': 'Foo\n```<y.txt>```',
            'y.txt': 'Bar\n```<x.txt>```',
        }
        with patch('builtins.open', mock_open()) as m:
            m.side_effect = _make_counting_mock_open(file_map)

            try:
                process_backtick_includes(
                    '```<x.txt>```',
                    recursive=False,
                )
            except ValueError as exc:
                assert "circular" in str(exc).lower()
                return
            except _InfiniteLoopDetected:
                pytest.fail(
                    "process_backtick_includes() looped 20+ times on circular A→B→A "
                    "includes in non-recursive mode. Bug #157: _seen is never populated "
                    "when recursive=False."
                )

            pytest.fail(
                "process_backtick_includes() returned without raising ValueError on "
                "circular A→B→A backtick includes."
            )

    def test_self_referencing_backtick_nonrecursive(self):
        """Self-referencing backtick include must be detected in non-recursive mode."""
        file_map = {
            'loop.txt': 'Data\n```<loop.txt>```',
        }
        with patch('builtins.open', mock_open()) as m:
            m.side_effect = _make_counting_mock_open(file_map)

            try:
                process_backtick_includes(
                    '```<loop.txt>```',
                    recursive=False,
                )
            except ValueError as exc:
                assert "circular" in str(exc).lower()
                return
            except _InfiniteLoopDetected:
                pytest.fail(
                    "process_backtick_includes() looped 20+ times on self-referencing "
                    "include in non-recursive mode. Bug #157: no cycle detection when "
                    "recursive=False."
                )

            pytest.fail(
                "process_backtick_includes() returned without raising ValueError on "
                "self-referencing backtick include."
            )

    # -- Regression guards: non-circular cases must still work --------------

    def test_linear_chain_nonrecursive_no_false_positive(self):
        """Linear A→B→C chain (no cycle) must resolve correctly in non-recursive mode."""
        file_map = {
            'top.txt': 'Top\n<include>mid.txt</include>',
            'mid.txt': 'Mid\n<include>leaf.txt</include>',
            'leaf.txt': 'Leaf',
        }
        with patch('builtins.open', mock_open()) as m:
            m.side_effect = _make_mock_open(file_map)

            result = process_include_tags(
                '<include>top.txt</include>',
                recursive=False,
            )

            assert 'Top' in result
            assert 'Mid' in result
            assert 'Leaf' in result
            # No include tags should remain
            assert '<include>' not in result

    def test_diamond_pattern_nonrecursive_no_false_positive(self):
        """Diamond pattern (same file included at same level) is NOT circular."""
        file_map = {
            'shared.txt': 'SharedContent',
        }
        with patch('builtins.open', mock_open()) as m:
            m.side_effect = _make_mock_open(file_map)

            result = process_include_tags(
                '<include>shared.txt</include>\n<include>shared.txt</include>',
                recursive=False,
            )

            # shared.txt included twice at the same level — that's fine
            assert result.count('SharedContent') == 2


# ---------------------------------------------------------------------------
# Issue #158: Additional tests for non-recursive circular include detection.
#
# These tests cover gaps identified in Step 6:
# 1. Top-level preprocess() wrapper (not just inner functions)
# 2. Three-file backtick cycle
# 3. Mixed XML + backtick circular includes
# 4. Circular + non-circular includes coexisting
# 5. Diamond pattern with actual content (regression guard)
# ---------------------------------------------------------------------------


class TestCircularIncludesNonRecursiveStep6:
    """Issue #158 Step 6: Additional non-recursive circular include tests."""

    def setup_method(self):
        set_pdd_path('/mock/path')

    def teardown_method(self):
        if _original_pdd_path is not None:
            os.environ['PDD_PATH'] = _original_pdd_path
        elif 'PDD_PATH' in os.environ:
            del os.environ['PDD_PATH']

    # -- Test 1: Top-level preprocess() with circular XML includes -----------

    def test_preprocess_circular_xml_nonrecursive(self):
        """Top-level preprocess() must detect circular XML includes in non-recursive mode.

        Previous tests called process_include_tags() directly. This test
        verifies that preprocess() (the public entry point) also detects
        cycles when recursive=False (the default).
        """
        file_map = {
            'a.txt': 'Alpha\n<include>b.txt</include>',
            'b.txt': 'Beta\n<include>a.txt</include>',
        }
        with patch('builtins.open', mock_open()) as m:
            m.side_effect = _make_counting_mock_open(file_map)

            try:
                result = preprocess(
                    '<include>a.txt</include>',
                    recursive=False,
                    double_curly_brackets=False,
                )
            except ValueError as exc:
                # Correct behavior — cycle detected at top level
                assert "circular" in str(exc).lower(), (
                    f"ValueError raised but not about circular includes: {exc}"
                )
                return
            except _InfiniteLoopDetected:
                pytest.fail(
                    "preprocess() looped 20+ times on circular A→B→A XML includes "
                    "in non-recursive mode. Bug #158: _seen is never populated "
                    "when recursive=False, so the while loop never converges."
                )

            pytest.fail(
                "preprocess() returned without raising ValueError on circular "
                "A→B→A XML includes in non-recursive mode."
            )

    # -- Test 2: Top-level preprocess() with circular backtick includes ------

    def test_preprocess_circular_backtick_nonrecursive(self):
        """Top-level preprocess() must detect circular backtick includes in non-recursive mode."""
        file_map = {
            'p.txt': 'Ping\n```<q.txt>```',
            'q.txt': 'Pong\n```<p.txt>```',
        }
        with patch('builtins.open', mock_open()) as m:
            m.side_effect = _make_counting_mock_open(file_map)

            try:
                result = preprocess(
                    '```<p.txt>```',
                    recursive=False,
                    double_curly_brackets=False,
                )
            except ValueError as exc:
                assert "circular" in str(exc).lower(), (
                    f"ValueError raised but not about circular includes: {exc}"
                )
                return
            except _InfiniteLoopDetected:
                pytest.fail(
                    "preprocess() looped 20+ times on circular A→B→A backtick "
                    "includes in non-recursive mode. Bug #158: _seen is never "
                    "populated when recursive=False."
                )

            pytest.fail(
                "preprocess() returned without raising ValueError on circular "
                "A→B→A backtick includes in non-recursive mode."
            )

    # -- Test 3: Three-file backtick cycle -----------------------------------

    def test_three_file_cycle_backtick_nonrecursive(self):
        """A→B→C→A three-file cycle via backtick includes must be detected.

        The existing tests only cover three-file cycles for XML <include> tags.
        Backtick includes use a separate function (process_backtick_includes)
        that has the same bug.
        """
        file_map = {
            'r.txt': 'Red\n```<g.txt>```',
            'g.txt': 'Green\n```<b.txt>```',
            'b.txt': 'Blue\n```<r.txt>```',
        }
        with patch('builtins.open', mock_open()) as m:
            m.side_effect = _make_counting_mock_open(file_map)

            try:
                process_backtick_includes(
                    '```<r.txt>```',
                    recursive=False,
                )
            except ValueError as exc:
                assert "circular" in str(exc).lower()
                return
            except _InfiniteLoopDetected:
                pytest.fail(
                    "process_backtick_includes() looped 20+ times on A→B→C→A "
                    "backtick cycle in non-recursive mode. Bug #158: _seen is "
                    "never populated when recursive=False."
                )

            pytest.fail(
                "process_backtick_includes() returned without raising ValueError on "
                "A→B→C→A backtick cycle in non-recursive mode."
            )

    # -- Test 4: Circular + non-circular includes in same text ---------------

    def test_mixed_circular_and_noncircular_nonrecursive(self):
        """Text with both circular and non-circular includes must detect the cycle.

        The non-circular include (safe.txt) should NOT prevent the circular
        pair (cyc1.txt ↔ cyc2.txt) from being detected.
        """
        file_map = {
            'safe.txt': 'SafeContent',
            'cyc1.txt': '<include>cyc2.txt</include>',
            'cyc2.txt': '<include>cyc1.txt</include>',
        }
        with patch('builtins.open', mock_open()) as m:
            m.side_effect = _make_counting_mock_open(file_map)

            try:
                process_include_tags(
                    '<include>safe.txt</include>\n<include>cyc1.txt</include>',
                    recursive=False,
                )
            except ValueError as exc:
                assert "circular" in str(exc).lower()
                return
            except _InfiniteLoopDetected:
                pytest.fail(
                    "process_include_tags() looped 20+ times when text contains "
                    "both circular and non-circular includes. Bug #158: the "
                    "non-circular include doesn't prevent the cycle from hanging."
                )

            pytest.fail(
                "process_include_tags() returned without raising ValueError when "
                "text contains both circular (cyc1↔cyc2) and non-circular (safe) includes."
            )

    # -- Test 5: Diamond pattern with content (regression guard) -------------

    def test_diamond_with_content_nonrecursive_no_false_positive(self):
        """Diamond pattern with real content must resolve correctly, not false-positive.

        A→B, A→C, B→D, C→D is NOT circular. D is included twice (via B and C),
        which is fine. This tests that the cycle detection doesn't over-trigger
        on legitimate diamond-shaped include graphs with actual file content.
        """
        file_map = {
            'a.txt': '<include>b.txt</include>\n<include>c.txt</include>',
            'b.txt': 'BContent\n<include>d.txt</include>',
            'c.txt': 'CContent\n<include>d.txt</include>',
            'd.txt': 'SharedLeaf',
        }
        with patch('builtins.open', mock_open()) as m:
            m.side_effect = _make_mock_open(file_map)

            result = process_include_tags(
                '<include>a.txt</include>',
                recursive=False,
            )

            assert 'BContent' in result, "B's content should be present"
            assert 'CContent' in result, "C's content should be present"
            assert result.count('SharedLeaf') == 2, (
                "D is included twice via B and C — that's fine, not circular"
            )
            assert '<include>' not in result, "All include tags should be resolved"
