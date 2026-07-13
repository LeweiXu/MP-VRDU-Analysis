"""Composes the T/TL/TLV/V representation for a cell (cost-ordered, parser text,
no bounding boxes)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from config import REPRESENTATION_LADDER
from schema import Modality, Page, Part, Payload, TextPart
from tools.parser import DEFAULT_PARSER, parser_markdown
from tools.text import embedded_text
from tools.visual import visual_channel


# The four cost-ordered rungs. The ladder is not cumulative: TL's parser text
# replaces T's embedded text rather than adding to it, and there is no separate
# layout channel (the "L" is historical). V is the image-only reference point.
RUNGS: tuple[str, ...] = REPRESENTATION_LADDER


class Representation(ABC):
    """Compose rendered pages into a reasoner `Payload`."""

    #: The ladder rung this composer implements (`T`, `TL`, `TLV`, `V`).
    modality: Modality

    @abstractmethod
    def build(self, pages: Sequence[Page]) -> Payload:
        """Encode the given pages into a `Payload` for the reasoner."""


def _text_block(label: str, chunks: Sequence[str]) -> list[Part]:
    """Join per-page channel strings into one labelled text part (or nothing)."""

    body = "\n\n".join(chunk for chunk in chunks if chunk).strip()
    return [TextPart(f"[{label}]\n{body}")] if body else []


class LadderRepresentation(Representation):
    """One of the four cost-ordered rungs.

    T uses cheap embedded text; TL and TLV use the parser's markdown text (read
    from the warmed disk cache) instead; TLV and V attach the page images.
    """

    def __init__(self, modality: str, parser_tool: str = DEFAULT_PARSER, dpi: int = 200) -> None:
        self.modality = modality
        self.parser_tool = parser_tool
        self.dpi = dpi

    def build(self, pages: Sequence[Page]) -> Payload:
        parts: list[Part] = []
        if self.modality == "T":
            parts += _text_block("text", embedded_text(pages))
        elif self.modality == "TL":
            parts += _text_block("text", parser_markdown(pages, self.parser_tool, self.dpi))
        elif self.modality == "TLV":
            parts += _text_block("text", parser_markdown(pages, self.parser_tool, self.dpi))
            parts += list(visual_channel(pages))
        elif self.modality == "V":
            parts += list(visual_channel(pages))
        return Payload(self.modality, tuple(parts))


def get_representation(modality: Modality, parser_tool: str = DEFAULT_PARSER, dpi: int = 200) -> Representation:
    """Return a representation composer for a ladder rung (T/TL/TLV/V)."""

    name = str(modality)
    if name not in RUNGS:
        raise KeyError(f"unknown representation {name!r}; use one of {RUNGS}")
    return LadderRepresentation(name, parser_tool, dpi)
