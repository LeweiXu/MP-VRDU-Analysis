"""MinerU parser (Stage 5 — layout/table-aware, ML-based; Apache-2.0).

Strong on scientific/financial + CJK; emits Markdown/JSON with structure. Heavy
(PaddleOCR backend, models). Lazy-imported; only constructed when a config asks
for parser: mineru. The MinerU API has shifted across versions, so this wraps
the common `magic_pdf` pipeline defensively and degrades with a clear error.
"""

from __future__ import annotations

import re
from pathlib import Path

from .base import Parser, ParsedPage, _markdown_sections


class MinerUParser(Parser):
    name = "mineru"
    structure_aware = True

    def __init__(self):
        # Defer heavy import to construction so importing this module is cheap.
        try:
            import magic_pdf  # noqa: F401
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(
                "MinerU not installed. `pip install magic-pdf[full]` (heavy; "
                "build the env on Kaya under /group). See context.md §12."
            ) from e

    def _markdown_by_page(self, pdf_path: str | Path) -> list[str]:
        """Run MinerU and return per-page markdown.

        MinerU emits a single markdown stream with page breaks; we split on its
        page-separator convention. Implementation is version-tolerant.
        """
        from magic_pdf.data.dataset import PymuDocDataset  # type: ignore
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze  # type: ignore

        data = Path(pdf_path).read_bytes()
        ds = PymuDocDataset(data)
        infer = ds.apply(doc_analyze, ocr=True)
        md = infer.pipe_ocr_mode(None).get_markdown("")
        # Split into pages on form-feed / explicit page markers if present.
        parts = re.split(r"\f|\n-{3,}\s*page\s*break\s*-{3,}\n", md, flags=re.I)
        return parts if len(parts) > 1 else [md]

    def parse_document(self, pdf_path: str | Path) -> list[ParsedPage]:
        mds = self._markdown_by_page(pdf_path)
        return [ParsedPage(page_index=i, text=md, markdown=md,
                           sections=_markdown_sections(md))
                for i, md in enumerate(mds)]

    def parse_page(self, pdf_path: str | Path, page_index: int) -> str:
        mds = self._markdown_by_page(pdf_path)
        return mds[page_index] if page_index < len(mds) else ""
