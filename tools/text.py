"""Cheap embedded-text extraction for the T channel via PyMuPDF."""

from __future__ import annotations

from collections.abc import Sequence

from schema import Page


def embedded_text(pages: Sequence[Page]) -> tuple[str, ...]:
    """Return each page's embedded PDF text (empty string for a scanned page).

    This is the cheap floor of the ladder: it reads the text layer PyMuPDF
    already extracted during rendering, so it costs nothing extra and returns
    nothing useful on a scanned document by design.
    """

    return tuple(page.text for page in pages)
