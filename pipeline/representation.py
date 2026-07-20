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


# The cost-ordered rungs. The ladder is not cumulative: TL's parser text replaces T's
# embedded text rather than adding to it, and there is no separate layout channel (the
# "L" is historical). V is the image-only reference point, and TLVi is TLV's per-page
# interleaved ordering at the same cost.
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


def _interleaved(pages: Sequence[Page], chunks: Sequence[str]) -> list[Part]:
    """Emit each page's text immediately followed by that page's image.

    The merged form (`_text_block` plus every image after it) gives the model no way
    to tell which text belongs to which image: the per-page chunks are concatenated
    with blank lines and the images carry no markers, so on a multi-page cell the
    association is unrecoverable. Here each page contributes a `[page N]`-headed text
    part and then its own image, so position alone carries the pairing. `N` is the
    1-based document page number, matching what is printed on the page itself.

    Pairing is positional and `visual_channel` is strictly one part per page (it
    raises rather than skipping a page without an image), so the zip is total;
    `strict` makes any future drift fail loudly instead of silently misaligning.
    """

    images = visual_channel(pages)
    parts: list[Part] = []
    for page, chunk, image in zip(pages, chunks, images, strict=True):
        header = f"[page {page.index + 1}]"
        body = (chunk or "").strip()
        parts.append(TextPart(f"{header}\n{body}" if body else header))
        parts.append(image)
    return parts


class LadderRepresentation(Representation):
    """One of the cost-ordered rungs.

    T uses cheap embedded text; TL, TLV and TLVi use the parser's markdown text (read
    from the warmed disk cache) instead; TLV, TLVi and V attach the page images. TLV
    emits one merged text block then every image; TLVi emits page text and that page's
    image in turn.
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
        elif self.modality == "TLVi":
            parts += _interleaved(pages, parser_markdown(pages, self.parser_tool, self.dpi))
        elif self.modality == "V":
            parts += list(visual_channel(pages))
        return Payload(self.modality, tuple(parts))


def get_representation(modality: Modality, parser_tool: str = DEFAULT_PARSER, dpi: int = 200) -> Representation:
    """Return a representation composer for a ladder rung (T/TL/TLV/V)."""

    name = str(modality)
    if name not in RUNGS:
        raise KeyError(f"unknown representation {name!r}; use one of {RUNGS}")
    return LadderRepresentation(name, parser_tool, dpi)
