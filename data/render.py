"""Render PDF pages and extract line-level embedded text spans.

Purpose:
    Converts source PDFs into the `schema.Page` objects consumed by
    representation composers and retrievers. It owns page-count checks,
    deterministic render-cache paths, PNG generation, and PyMuPDF text-span
    extraction.

Pipeline role:
    The orchestrator resolves `Question` + `PageSet` selections into rendered
    pages through this module. Tool implementations then read the rendered image
    path and embedded spans from each `Page`.

Arguments:
    None. This module is import-only; callers pass PDF paths, page indices,
    cache directories, and DPI to `render_pdf()` or `render_question_pages()`.
"""

from __future__ import annotations

import re
from pathlib import Path

from config import DEFAULT_PATHS
from data.loader import resolve_pdf
from schema import Page, Question, TextSpan


def safe_stem(name: str) -> str:
    """Return a filesystem-safe stem while keeping it readable for debugging."""

    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "document"


def render_cache_dir(pdf_path: Path, cache_dir: Path | None = None, dpi: int = 144) -> Path:
    """Return the deterministic render directory for one PDF and DPI."""

    root = Path(cache_dir or DEFAULT_PATHS.cache_dir)
    return root / "renders" / f"{safe_stem(pdf_path.name)}__dpi{dpi}"


def pdf_page_count(pdf_path: Path) -> int:
    """Return the number of pages in a PDF."""

    import fitz

    with fitz.open(pdf_path) as doc:
        return doc.page_count


def extract_text_spans(page) -> tuple[TextSpan, ...]:
    """Extract line-level text spans from a PyMuPDF page."""

    spans: list[TextSpan] = []
    text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            line_text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if not line_text:
                continue
            bbox = line.get("bbox")
            spans.append(
                TextSpan(
                    text=line_text,
                    bbox=tuple(float(value) for value in bbox) if bbox else None,
                )
            )
    return tuple(spans)


def render_pdf(
    pdf_path: Path,
    page_indices: tuple[int, ...] | list[int] | None = None,
    cache_dir: Path | None = None,
    dpi: int = 144,
    render_images: bool = True,
    extract_text: bool = True,
) -> list[Page]:
    """Render/extract selected zero-based pages from a PDF.

    Images are cached as PNGs under `results/cache/renders`. Text spans are
    extracted on each call because they are cheap and keep the cache format
    simple.
    """

    import fitz

    pdf = Path(pdf_path)
    if not pdf.is_file():
        raise FileNotFoundError(pdf)

    out: list[Page] = []
    target_dir = render_cache_dir(pdf, cache_dir, dpi)
    if render_images:
        target_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(pdf) as doc:
        indices = tuple(range(doc.page_count)) if page_indices is None else tuple(int(i) for i in page_indices)
        for index in indices:
            if index < 0 or index >= doc.page_count:
                raise IndexError(f"page index {index} out of range for {pdf} with {doc.page_count} pages")
            page = doc.load_page(index)
            image_path: Path | None = None
            if render_images:
                image_path = target_dir / f"page_{index:04d}.png"
                if not image_path.exists():
                    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
                    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                    pixmap.save(str(image_path))
            spans = extract_text_spans(page) if extract_text else ()
            out.append(
                Page(
                    doc_id=pdf.name,
                    index=index,
                    pdf_path=pdf,
                    image_path=image_path,
                    text_spans=spans,
                )
            )
    return out


def render_question_pages(
    question: Question,
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
    page_indices: tuple[int, ...] | list[int] | None = None,
    dpi: int = 144,
) -> list[Page]:
    """Resolve and render pages for a question.

    If `page_indices` is omitted, gold evidence pages are rendered. For native
    unanswerable questions with no gold pages, page 0 is rendered as a cheap
    document sanity check.
    """

    pdf = resolve_pdf(question.doc_id, data_dir)
    indices = tuple(page_indices) if page_indices is not None else question.evidence_pages
    if not indices:
        indices = (0,)
    return render_pdf(pdf, indices, cache_dir=cache_dir, dpi=dpi)


def validate_gold_pages(question: Question, data_dir: Path | None = None) -> bool:
    """Return whether all gold evidence pages are within the resolved PDF."""

    if not question.evidence_pages:
        return True
    pdf = resolve_pdf(question.doc_id, data_dir)
    count = pdf_page_count(pdf)
    return all(0 <= page < count for page in question.evidence_pages)
