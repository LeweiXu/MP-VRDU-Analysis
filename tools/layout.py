"""Layout extraction tools for markdown structure and bounding-box geometry.

Stage 3 placeholder: `layout_channel` just echoes the page text (no structure
recovery yet), so the `TL` composer is runnable end to end. Stage M2 (v3)
replaces this with Marker text + serialized bbox JSON as the primary `T+L`
source, behind the same return type, without touching the composer. The
Docling/PP-Structure paths stay available for the appendix parser-swap. (This
reverses the v1 Docling-primary note above; Marker-vs-Docling primary is
confirmed at the M1/M2 checkpoint.)
"""

from __future__ import annotations

from collections.abc import Sequence

from schema import Page


def layout_channel(pages: Sequence[Page]) -> tuple[str, ...]:
    """Return per-page layout/structure text for the layout channel."""

    return tuple(page.text for page in pages)
