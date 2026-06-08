"""Chunking (cross-cutting; Stage 4 uses it, Stage 5 sub-study varies it).

Turns parsed pages into retrievable chunks. Each chunk remembers its source
0-based page so retrieval-recall stays page-based regardless of granularity.

Strategies:
- "page":    one chunk per page (the default).
- "chunk":   fixed-size word windows within each page (with overlap).
- "section": one chunk per markdown section; gracefully DEGRADES to "page" when
             the parser gave no structure (context.md / plan Stage 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base import ParsedPage


@dataclass
class Chunk:
    page_index: int          # 0-based source page (for recall)
    text: str
    section: Optional[str] = None


def _window(words: list[str], size: int, overlap: int):
    if size <= 0:
        yield words
        return
    step = max(1, size - overlap)
    for start in range(0, max(1, len(words)), step):
        piece = words[start:start + size]
        if piece:
            yield piece
        if start + size >= len(words):
            break


def chunk_pages(pages: list[ParsedPage], strategy: str = "page",
                chunk_words: int = 180, overlap: int = 30) -> list[Chunk]:
    if strategy == "page":
        return [Chunk(page_index=p.page_index, text=p.text) for p in pages]

    if strategy == "chunk":
        chunks: list[Chunk] = []
        for p in pages:
            words = p.text.split()
            if not words:
                chunks.append(Chunk(page_index=p.page_index, text=p.text))
                continue
            for piece in _window(words, chunk_words, overlap):
                chunks.append(Chunk(page_index=p.page_index, text=" ".join(piece)))
        return chunks

    if strategy == "section":
        chunks = []
        any_structure = False
        for p in pages:
            if p.sections:
                any_structure = True
                for heading, body in p.sections:
                    text = (f"{heading}\n{body}" if heading else body).strip()
                    if text:
                        chunks.append(Chunk(page_index=p.page_index, text=text,
                                            section=heading or None))
            else:
                chunks.append(Chunk(page_index=p.page_index, text=p.text))
        if not any_structure:
            # graceful degradation: no parser structure -> page-level
            return chunk_pages(pages, "page")
        return chunks

    raise ValueError(f"unknown chunking strategy {strategy!r}")
