"""Embedded-text and OCR extraction tools for the text channel.

Stage 3 placeholder: `text_channel` returns the embedded text PyMuPDF already
pulled during rendering, one string per page. Stage M2 (v3) replaces this with
real `embedded()` (PyMuPDF) and `ocr()` (PaddleOCR PP-OCRv5) variants behind the
same return type, so the `T`/`TL` composers keep calling `text_channel`
unchanged. Note: v3 makes Marker (in `tools/layout.py`) the primary text source
for the ladder; PyMuPDF embedded text is the appendix parser-swap.
"""

from __future__ import annotations

from collections.abc import Sequence

from schema import Page


def text_channel(pages: Sequence[Page]) -> tuple[str, ...]:
    """Return per-page text for the text channel."""

    return tuple(page.text for page in pages)
