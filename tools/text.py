"""Provide embedded-text and OCR extraction tools for text channels.

Purpose:
    Keeps the text extraction variants separate from representation composers:
    Marker text is primary for digital-born documents in the v3 ladder, while
    PaddleOCR PP-OCRv5 supplies the text channel for documents labelled as
    scanned in `annotations/doc_labels.csv`. PyMuPDF embedded text remains the
    cheap fallback.

Pipeline role:
    `text_channel(pages)` routes each document to Marker or OCR while preserving
    the frozen representation interface and the existing `T`/`TL`/`TLV` rungs.
    `embedded()` and `ocr()` are also used by tool smoke tests and parser/OCR
    comparisons.

Arguments:
    None at the command line. Public call inputs are `pages` sequences of
    `schema.Page` objects; `ocr()` also accepts an optional injectable OCR
    `engine` and `allow_embedded_fallback` flag.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import DEFAULT_PATHS, ROOT
from schema import Page

ANNOTATION_SHEET = ROOT / "annotations" / "doc_labels.csv"
SCAN_LABELS = {"scanned", "digital"}


def embedded(pages: Sequence[Page]) -> tuple[str, ...]:
    """Return per-page embedded PDF text using PyMuPDF spans from rendering."""

    return tuple(page.text for page in pages)


def _paddle_ocr_engine() -> Any:
    """Construct the default PaddleOCR PP-OCRv5 English engine."""

    from paddleocr import PaddleOCR

    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en",
    )


def _safe_stem(name: str) -> str:
    """Filesystem-safe PDF stem for text-tool cache files."""

    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "document"


def _page_cache_root(page: Page) -> Path:
    """Return the shared cache root derived from a rendered page image."""

    if page.image_path is not None:
        try:
            # <root>/renders/<stem>__dpiN/page_XXXX.png -> parents[2] == <root>
            return Path(page.image_path).parents[2]
        except IndexError:
            pass
    return Path(DEFAULT_PATHS.cache_dir)


def _render_dpi_tag(page: Page) -> str:
    """Return a stable DPI tag for an OCR cache key."""

    if page.image_path is None:
        return "dpiunknown"
    render_dir = Path(page.image_path).parent.name
    match = re.search(r"__dpi(\d+)$", render_dir)
    return f"dpi{match.group(1)}" if match else "dpiunknown"


def _ocr_cache_file(page: Page) -> Path:
    """Disk path for one page's cached OCR text artifact."""

    stem = _safe_stem(Path(page.pdf_path).name)
    return _page_cache_root(page) / "ocr" / f"{stem}__{_render_dpi_tag(page)}__p{page.index:04d}.txt"


def _read_ocr_cache(page: Page) -> str | None:
    """Return cached OCR text for a page, or None on miss/error."""

    path = _ocr_cache_file(page)
    if not path.exists():
        return None
    try:
        return path.read_text()
    except OSError:
        return None


def _write_ocr_cache(page: Page, value: str) -> None:
    """Persist one OCR text artifact (best effort; never raises)."""

    path = _ocr_cache_file(page)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    except OSError:
        pass


def _object_to_plain(value: Any) -> Any:
    """Best-effort conversion of Paddle result objects to plain containers."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {key: _object_to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_object_to_plain(item) for item in value]
    for attr in ("json", "res"):
        if hasattr(value, attr):
            try:
                return _object_to_plain(getattr(value, attr))
            except Exception:
                pass
    for method in ("to_dict", "dict", "model_dump"):
        if hasattr(value, method):
            try:
                return _object_to_plain(getattr(value, method)())
            except Exception:
                pass
    if hasattr(value, "__dict__"):
        return _object_to_plain(vars(value))
    return str(value)


def _collect_ocr_texts(value: Any) -> list[str]:
    """Collect recognised text strings from flexible PaddleOCR result shapes."""

    plain = _object_to_plain(value)
    texts: list[str] = []
    if isinstance(plain, str):
        stripped = plain.strip()
        return [stripped] if stripped else []
    if isinstance(plain, dict):
        for key in ("rec_texts", "text", "texts", "transcription"):
            item = plain.get(key)
            if isinstance(item, str) and item.strip():
                texts.append(item.strip())
            elif isinstance(item, (list, tuple)):
                texts.extend(str(x).strip() for x in item if str(x).strip())
        for item in plain.values():
            texts.extend(_collect_ocr_texts(item))
    elif isinstance(plain, (list, tuple)):
        for item in plain:
            texts.extend(_collect_ocr_texts(item))
    return list(dict.fromkeys(texts))


def _predict_ocr(engine: Any, image_path: Path) -> Any:
    """Run one PaddleOCR-compatible engine on an image path."""

    if hasattr(engine, "predict"):
        return engine.predict(str(image_path))
    if hasattr(engine, "ocr"):
        return engine.ocr(str(image_path))
    if callable(engine):
        return engine(str(image_path))
    raise TypeError("OCR engine must expose predict(), ocr(), or be callable")


def ocr(
    pages: Sequence[Page],
    *,
    engine: Any | None = None,
    allow_embedded_fallback: bool = True,
) -> tuple[str, ...]:
    """Return per-page OCR text using PaddleOCR PP-OCRv5.

    `engine` is injectable so tests can exercise result parsing without loading
    the heavy Paddle models. Real smoke runs use the default PaddleOCR engine.
    Successful OCR output is cached per (page, render DPI), so the parse pre-pass
    warms scanned-document text before the reasoner loads.
    """

    ocr_engine = engine
    out: list[str] = []
    for page in pages:
        if page.image_path is None:
            raise ValueError(f"page {page.index} has no image_path for OCR")
        cached = _read_ocr_cache(page)
        if cached is not None:
            out.append(cached)
            continue
        if ocr_engine is None:
            ocr_engine = _paddle_ocr_engine()
        texts = _collect_ocr_texts(_predict_ocr(ocr_engine, page.image_path))
        text = "\n".join(texts).strip()
        cache_ok = bool(text)
        if not text and allow_embedded_fallback:
            text = page.text.strip()
            cache_ok = False
        if cache_ok:
            _write_ocr_cache(page, text)
        out.append(text)
    return tuple(out)


@lru_cache(maxsize=1)
def _annotation_scan_labels(path: str) -> dict[str, str]:
    """Load document scan labels from the annotation sheet.

    Human `scan_label` values win when present; otherwise the auto-seeded
    `auto_scan` value is used. Invalid or blank values are ignored.
    """

    sheet = Path(path)
    if not sheet.exists():
        return {}
    labels: dict[str, str] = {}
    try:
        with sheet.open(newline="") as handle:
            for row in csv.DictReader(handle):
                doc_id = (row.get("doc_id") or "").strip()
                if not doc_id:
                    continue
                label = (row.get("scan_label") or "").strip() or (row.get("auto_scan") or "").strip()
                if label in SCAN_LABELS:
                    labels[doc_id] = label
    except OSError:
        return {}
    return labels


def document_scan_label(page: Page, *, annotation_sheet: Path | None = None) -> str:
    """Return `scanned` or `digital` for a rendered page's document.

    The annotation sheet is authoritative. If it has no row for the document,
    fall back to the embedded-text heuristic used to seed that sheet.
    """

    labels = _annotation_scan_labels(str(annotation_sheet or ANNOTATION_SHEET))
    label = labels.get(page.doc_id)
    if label in SCAN_LABELS:
        return label

    from data.render import classify_scanned

    return classify_scanned(Path(page.pdf_path)).label


def text_channel(pages: Sequence[Page], *, ocr_engine: Any | None = None) -> tuple[str, ...]:
    """Return the primary per-page text channel for the v3 ladder.

    Digital-born documents use Marker text. Scanned documents use PaddleOCR text,
    as labelled by `annotations/doc_labels.csv` (`scan_label` if filled, else
    `auto_scan`).
    """

    from tools.layout import marker_text

    if not pages:
        return ()
    if all(document_scan_label(page) == "scanned" for page in pages):
        return ocr(pages, engine=ocr_engine)
    if all(document_scan_label(page) == "digital" for page in pages):
        return marker_text(pages)

    out: list[str] = []
    for page in pages:
        if document_scan_label(page) == "scanned":
            out.extend(ocr((page,), engine=ocr_engine))
        else:
            out.extend(marker_text((page,)))
    return tuple(out)
