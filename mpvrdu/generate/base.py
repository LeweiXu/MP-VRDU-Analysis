"""Generator interface + shared input builder (Stage 2/3).

`InputBuilder` turns a Selection into the (images, text) a generator reads,
honouring the generation modality. Images are passed as PNG file PATHS (kept
dependency-light; a real VLM loads them lazily). Text is the parsed text of the
selected pages, joined with page markers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config import RunConfig
from ..data.dataset import Document, Question
from ..data.render import render_pages
from ..represent.base import get_parser
from ..retrieve.base import Selection


@dataclass
class GeneratorInput:
    question: str
    image_paths: list[str] = field(default_factory=list)
    text: Optional[str] = None
    n_pages: int = 0


class InputBuilder:
    """Builds generator inputs from selected pages per the run config."""

    def __init__(self, cfg: RunConfig):
        self.modality = cfg.generation.modality
        self.dpi = cfg.representation.dpi
        self.parser_name = cfg.representation.parser
        self._parser = None  # lazy: text modalities only

    @property
    def parser(self):
        if self._parser is None:
            self._parser = get_parser(self.parser_name)
        return self._parser

    def build(self, question: Question, document: Document,
              selection: Selection) -> GeneratorInput:
        pages = selection.page_indices
        image_paths: list[str] = []
        text: Optional[str] = None

        if self.modality in {"image", "both"} and pages:
            rendered = render_pages(document.pdf_path, pages, dpi=self.dpi,
                                    doc_id=document.doc_id)
            image_paths = [str(r.path) for r in rendered]

        if self.modality in {"text", "both"}:
            parts = []
            for p in pages:
                page_text = self.parser.parse_page(document.pdf_path, p)
                parts.append(f"[page {p + 1}]\n{page_text}")
            text = "\n\n".join(parts) if parts else ""

        return GeneratorInput(question=question.question, image_paths=image_paths,
                              text=text, n_pages=len(pages))


class Generator(ABC):
    name: str = "base"

    @abstractmethod
    def answer(self, question: str, images: Optional[list[str]] = None,
               text: Optional[str] = None) -> str:
        ...

    def answer_input(self, gi: GeneratorInput) -> str:
        return self.answer(gi.question, images=gi.image_paths, text=gi.text)
