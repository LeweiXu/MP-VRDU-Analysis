"""Parser interface + registry (Stage 2/5).

`parse_document` returns one `ParsedPage` per 0-based page, carrying plain text
plus optional markdown and section structure. Text retrievers (Stage 4) and the
text/both generation modalities consume this; the Stage-5 sub-study swaps the
parser while everything downstream stays identical.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass
class ParsedPage:
    page_index: int                 # 0-based
    text: str
    markdown: Optional[str] = None
    # (heading, body) pairs when the parser is structure-aware; else None.
    sections: Optional[list[tuple[str, str]]] = None


class Parser(ABC):
    name: str = "base"
    structure_aware: bool = False   # True if it can emit sections

    @abstractmethod
    def parse_page(self, pdf_path: str | Path, page_index: int) -> str:
        """Return the plain text of a single 0-based page."""

    def parse_document(self, pdf_path: str | Path) -> list[ParsedPage]:
        """Default: per-page plain text, no structure. Override for richer output."""
        import fitz

        with fitz.open(pdf_path) as doc:
            n = doc.page_count
        return [ParsedPage(page_index=p, text=self.parse_page(pdf_path, p))
                for p in range(n)]

    def parse_pages(self, pdf_path: str | Path, page_indices) -> list[str]:
        return [self.parse_page(pdf_path, p) for p in page_indices]


class PyMuPDFParser(Parser):
    """Native text-layer extraction via PyMuPDF (fast, born-digital PDFs).

    NOTE: PyMuPDF is AGPL-3.0 (context.md §10) — flagged for any code release.
    """

    name = "pymupdf"

    def parse_page(self, pdf_path: str | Path, page_index: int) -> str:
        import fitz

        with fitz.open(pdf_path) as doc:
            return doc.load_page(page_index).get_text("text")


class PyMuPDF4LLMParser(Parser):
    """PyMuPDF4LLM: native text layer serialised to Markdown (Stage-4 default).

    Markdown headings make section-aware chunking possible. Per-page markdown is
    produced by restricting `to_markdown` to a single page. Also AGPL-3.0.
    """

    name = "pymupdf4llm"
    structure_aware = True

    def parse_page(self, pdf_path: str | Path, page_index: int) -> str:
        return self._page_markdown(pdf_path, page_index)

    def _page_markdown(self, pdf_path: str | Path, page_index: int) -> str:
        import pymupdf4llm

        return pymupdf4llm.to_markdown(str(pdf_path), pages=[page_index],
                                       show_progress=False)

    def parse_document(self, pdf_path: str | Path) -> list[ParsedPage]:
        import fitz

        with fitz.open(pdf_path) as doc:
            n = doc.page_count
        pages = []
        for p in range(n):
            md = self._page_markdown(pdf_path, p)
            pages.append(ParsedPage(page_index=p, text=md, markdown=md,
                                    sections=_markdown_sections(md)))
        return pages


def _markdown_sections(md: str) -> Optional[list[tuple[str, str]]]:
    """Split markdown into (heading, body) sections on # headings."""
    import re

    lines = md.splitlines()
    sections: list[tuple[str, str]] = []
    heading = ""
    body: list[str] = []
    saw_heading = False
    for line in lines:
        m = re.match(r"^#{1,6}\s+(.*)", line.strip())
        if m:
            if saw_heading or body:
                sections.append((heading, "\n".join(body).strip()))
            heading = m.group(1).strip()
            body = []
            saw_heading = True
        else:
            body.append(line)
    if saw_heading or body:
        sections.append((heading, "\n".join(body).strip()))
    return sections if saw_heading else None


_REGISTRY: dict[str, Callable[[], Parser]] = {
    "pymupdf": PyMuPDFParser,
    "pymupdf4llm": PyMuPDF4LLMParser,
}


def register_parser(name: str, factory: Callable[[], Parser]) -> None:
    _REGISTRY[name] = factory


def get_parser(name: str) -> Parser:
    if name not in _REGISTRY:
        # lazy-register the heavy Stage-5 backends only when asked for
        if name == "mineru":
            from .mineru import MinerUParser

            register_parser("mineru", MinerUParser)
        elif name == "tesseract":
            from .tesseract import TesseractParser

            register_parser("tesseract", TesseractParser)
        else:
            raise ValueError(f"unknown parser {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()
