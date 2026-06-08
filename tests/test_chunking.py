"""Stage 5 — chunking strategies + graceful degradation."""

from mpvrdu.represent.base import ParsedPage
from mpvrdu.represent.chunking import chunk_pages


def _pages_plain():
    return [ParsedPage(0, "alpha beta gamma " * 100),
            ParsedPage(1, "delta epsilon")]


def test_page_chunking_one_per_page():
    chunks = chunk_pages(_pages_plain(), "page")
    assert len(chunks) == 2
    assert [c.page_index for c in chunks] == [0, 1]


def test_fixed_chunking_splits_long_page():
    chunks = chunk_pages(_pages_plain(), "chunk", chunk_words=50, overlap=10)
    # the 300-word page 0 must split into multiple chunks; page 1 stays one
    page0 = [c for c in chunks if c.page_index == 0]
    page1 = [c for c in chunks if c.page_index == 1]
    assert len(page0) > 1
    assert len(page1) == 1
    # every chunk remembers its source page (so recall stays page-based)
    assert all(c.page_index in (0, 1) for c in chunks)


def test_section_chunking_uses_structure():
    pages = [ParsedPage(0, "x", markdown="# H1\nbody one\n# H2\nbody two",
                        sections=[("H1", "body one"), ("H2", "body two")])]
    chunks = chunk_pages(pages, "section")
    assert len(chunks) == 2
    assert chunks[0].section == "H1"
    assert all(c.page_index == 0 for c in chunks)


def test_section_chunking_degrades_without_structure():
    # no sections -> falls back to page-level rather than failing
    chunks = chunk_pages(_pages_plain(), "section")
    assert len(chunks) == 2
    assert all(c.section is None for c in chunks)
