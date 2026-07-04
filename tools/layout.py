"""Extract Marker text and serialized bounding-box layout artifacts.

Purpose:
    Implements the v3 primary parser path for `T` and `T+L`. Marker supplies
    per-page text and bbox-bearing layout JSON; PyMuPDF fallback output keeps
    local tests and appendix parser-swap probes runnable before Marker is
    installed.

Pipeline role:
    `tools.text.text_channel()` calls `marker_text()`. `layout_channel()` calls
    `marker_bbox_json()` so `TL` and `TLV` payloads receive serialized layout
    strings without changing the frozen representation interface. Kaya smoke
    uses `allow_fallback=False` to prove the real Marker path works.

Arguments:
    None at the command line. Public call inputs are `pages` sequences of
    `schema.Page` objects; `marker_text()` and `marker_bbox_json()` also accept
    `allow_fallback` to control whether PyMuPDF fallback is permitted.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from schema import Page


class MarkerUnavailableError(RuntimeError):
    """Raised when Marker is required but cannot produce an artifact."""


def _jsonable(value: Any) -> Any:
    """Convert Pydantic/dataclass-like Marker output into JSONable objects."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    for method in ("model_dump", "dict", "to_dict"):
        if hasattr(value, method):
            try:
                return _jsonable(getattr(value, method)())
            except Exception:
                pass
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _strip_html(value: str) -> str:
    """Return a compact text view of a block's HTML/text content."""

    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


def _bbox_from_polygon(polygon: Any) -> list[float] | None:
    """Convert a Marker polygon into an axis-aligned bbox if possible."""

    if not isinstance(polygon, list) or not polygon:
        return None
    try:
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
    except (TypeError, ValueError, IndexError):
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _block_text(block: dict[str, Any]) -> str:
    """Extract the best short text field from one Marker block."""

    for key in ("text", "html", "markdown", "content"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return _strip_html(value)
    return ""


def _collect_marker_blocks(value: Any) -> list[dict[str, Any]]:
    """Flatten Marker JSON/chunks output to bbox-bearing block dictionaries."""

    plain = _jsonable(value)
    blocks: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            bbox = node.get("bbox") or _bbox_from_polygon(node.get("polygon"))
            has_content = bbox is not None or any(key in node for key in ("text", "html", "markdown"))
            if has_content:
                blocks.append(
                    {
                        "type": str(node.get("block_type") or node.get("type") or node.get("label") or "Block"),
                        "text": _block_text(node),
                        "bbox": [float(x) for x in bbox] if bbox else None,
                    }
                )
            for child_key in ("children", "blocks", "items"):
                visit(node.get(child_key))
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(plain)
    return blocks


def _fallback_layout(page: Page, *, source: str, error: str | None = None) -> dict[str, Any]:
    """Build a deterministic PyMuPDF line-level layout artifact for one page."""

    blocks = [
        {
            "type": "TextLine",
            "text": span.text,
            "bbox": [float(x) for x in span.bbox] if span.bbox else None,
        }
        for span in page.text_spans
        if span.text
    ]
    artifact: dict[str, Any] = {
        "source": source,
        "doc_id": page.doc_id,
        "pdf_path": str(page.pdf_path),
        "page_index": page.index,
        "blocks": blocks,
    }
    if error:
        artifact["fallback_error"] = error
    return artifact


def _marker_converter(output_format: str, page_index: int) -> Any:
    """Create a Marker PDF converter for one page and output format."""

    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    config_parser = ConfigParser(
        {
            "output_format": output_format,
            "page_range": str(page_index),
            "disable_image_extraction": True,
        }
    )
    return PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )


def _marker_render(page: Page, output_format: str) -> Any:
    """Run Marker on a single selected PDF page."""

    converter = _marker_converter(output_format, page.index)
    return converter(str(page.pdf_path))


def _marker_text_from_rendered(rendered: Any) -> str:
    """Extract markdown/text from Marker rendered output."""

    from marker.output import text_from_rendered

    text, _, _ = text_from_rendered(rendered)
    return str(text).strip()


def marker_text(pages: Sequence[Page], *, allow_fallback: bool = True) -> tuple[str, ...]:
    """Return Marker-extracted text for each page."""

    out: list[str] = []
    for page in pages:
        try:
            text = _marker_text_from_rendered(_marker_render(page, "markdown"))
        except Exception as exc:
            if not allow_fallback:
                raise MarkerUnavailableError(f"Marker text failed for {page.pdf_path} page {page.index}: {exc}") from exc
            text = page.text
        out.append(text.strip())
    return tuple(out)


def marker_bbox_json(pages: Sequence[Page], *, allow_fallback: bool = True) -> tuple[str, ...]:
    """Return serialized per-page bbox layout JSON from Marker."""

    out: list[str] = []
    for page in pages:
        try:
            rendered = _marker_render(page, "json")
            blocks = _collect_marker_blocks(rendered)
            artifact = {
                "source": "marker",
                "doc_id": page.doc_id,
                "pdf_path": str(page.pdf_path),
                "page_index": page.index,
                "blocks": blocks,
            }
            if not blocks:
                raise MarkerUnavailableError("Marker returned no JSON blocks")
        except Exception as exc:
            if not allow_fallback:
                raise MarkerUnavailableError(
                    f"Marker bbox JSON failed for {page.pdf_path} page {page.index}: {exc}"
                ) from exc
            artifact = _fallback_layout(page, source="pymupdf-fallback", error=str(exc))
        out.append(json.dumps(artifact, sort_keys=True))
    return tuple(out)


def docling_available() -> bool:
    """Return whether the appendix Docling parser-swap package imports."""

    try:
        import docling  # noqa: F401

        return True
    except Exception:
        return False


def layout_channel(pages: Sequence[Page]) -> tuple[str, ...]:
    """Return the primary per-page layout channel for the v3 ladder."""

    return marker_bbox_json(pages)
