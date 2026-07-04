"""Provide embedded-text and OCR extraction tools for text channels.

Purpose:
    Keeps the text extraction variants separate from representation composers:
    Marker text is primary for the v3 ladder, while PyMuPDF embedded text and
    PaddleOCR PP-OCRv5 support appendix parser swaps and scanned-page checks.

Pipeline role:
    `text_channel(pages)` delegates to `tools.layout.marker_text()` so `T` and
    `T+L` consume the primary Marker source through the frozen representation
    interface. `embedded()` and `ocr()` are used by tool smoke tests and later
    parser/OCR comparisons.

Arguments:
    None at the command line. Public call inputs are `pages` sequences of
    `schema.Page` objects; `ocr()` also accepts an optional injectable OCR
    `engine` and `allow_embedded_fallback` flag.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from schema import Page


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
    """

    ocr_engine = engine or _paddle_ocr_engine()
    out: list[str] = []
    for page in pages:
        if page.image_path is None:
            raise ValueError(f"page {page.index} has no image_path for OCR")
        texts = _collect_ocr_texts(_predict_ocr(ocr_engine, page.image_path))
        text = "\n".join(texts).strip()
        if not text and allow_embedded_fallback:
            text = page.text.strip()
        out.append(text)
    return tuple(out)


def text_channel(pages: Sequence[Page]) -> tuple[str, ...]:
    """Return the primary per-page text channel for the v3 ladder."""

    from tools.layout import marker_text

    return marker_text(pages)
