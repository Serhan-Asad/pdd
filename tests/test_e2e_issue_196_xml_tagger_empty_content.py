"""
E2E Test for Issue #196: xml_tagger wraps empty string in duplicate tags

Bug: wrap_xml("test_tag", "") produces "<test_tag></test_tag></test_tag>"
     instead of "<test_tag></test_tag>".

This E2E test exercises wrap_xml through realistic usage patterns that a
consumer of the pdd.xml_tagger module would perform — building nested XML
structures, composing multiple tags, and using wrap_xml output as input to
further processing. Unlike the unit tests which check individual cases,
these tests verify that the bug cascades correctly through composed operations.
"""

import pytest


class TestE2EIssue196EmptyContentXmlWrapping:
    """E2E tests for Issue #196: wrap_xml duplicate closing tags on empty content."""

    def test_e2e_import_and_empty_wrap(self):
        """Full import-path test: pdd.xml_tagger.wrap_xml handles empty content.

        Simulates what a consumer module would do: import wrap_xml from
        pdd.xml_tagger and use it to wrap empty content.
        """
        from pdd.xml_tagger import wrap_xml

        result = wrap_xml("test_tag", "")
        # The bug produces "<test_tag></test_tag></test_tag>" (duplicate closing tag)
        assert result == "<test_tag></test_tag>", f"Got: {result}"
        # Verify no duplicate closing tags
        assert result.count("</test_tag>") == 1, (
            f"Duplicate closing tags detected: {result}"
        )

    def test_e2e_nested_xml_with_empty_inner(self):
        """Compose wrap_xml calls: outer tag wrapping an empty inner tag.

        This is a realistic pattern — building nested XML where an inner
        element has no content. The duplicate closing tag bug would produce
        malformed nesting.
        """
        from pdd.xml_tagger import wrap_xml

        inner = wrap_xml("inner", "")
        outer = wrap_xml("outer", inner)
        assert outer == "<outer><inner></inner></outer>", f"Got: {outer}"

    def test_e2e_multiple_empty_tags_concatenated(self):
        """Build a document from multiple empty-content tags.

        If wrap_xml has the duplicate closing tag bug, concatenating several
        empty-wrapped tags produces increasingly malformed XML.
        """
        from pdd.xml_tagger import wrap_xml

        parts = [
            wrap_xml("header", ""),
            wrap_xml("body", "some content"),
            wrap_xml("footer", ""),
        ]
        document = "\n".join(parts)
        assert "<header></header>" in document, f"Header malformed in: {document}"
        assert "<footer></footer>" in document, f"Footer malformed in: {document}"
        assert document.count("</header>") == 1, f"Duplicate header closing: {document}"
        assert document.count("</footer>") == 1, f"Duplicate footer closing: {document}"

    def test_e2e_wrap_xml_round_trip_parsing(self):
        """Verify wrap_xml output for empty content is valid XML that can be parsed.

        A duplicate closing tag like <tag></tag></tag> is not well-formed XML
        and will fail to parse.
        """
        from pdd.xml_tagger import wrap_xml
        import xml.etree.ElementTree as ET

        result = wrap_xml("valid_tag", "")
        # If there's a duplicate closing tag, ET.fromstring will raise ParseError
        try:
            elem = ET.fromstring(result)
        except ET.ParseError as exc:
            pytest.fail(
                f"wrap_xml produced invalid XML for empty content: {result!r} — {exc}"
            )
        assert elem.tag == "valid_tag"
        assert elem.text is None or elem.text == ""
