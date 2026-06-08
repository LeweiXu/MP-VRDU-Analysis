"""Tesseract OCR parser (Stage 5 — the OCR comparison point).

Works from RENDERED page images, so it handles scanned/raster pages that the
native-text-layer parsers cannot. Noisier than born-digital extraction. Lazy
imports keep the light env clean; needs `pytesseract` + the tesseract binary.
"""

from __future__ import annotations

from pathlib import Path

from ..data.render import DEFAULT_DPI, render_page
from .base import Parser


class TesseractParser(Parser):
    name = "tesseract"
    structure_aware = False

    def __init__(self, dpi: int = DEFAULT_DPI):
        self.dpi = dpi

    def parse_page(self, pdf_path: str | Path, page_index: int) -> str:
        import pytesseract
        from PIL import Image

        rp = render_page(pdf_path, page_index, dpi=self.dpi)
        with Image.open(rp.path) as img:
            return pytesseract.image_to_string(img)
