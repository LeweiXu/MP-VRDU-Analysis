"""Representation interfaces for text, layout, text-layout-visual, and visual payload composers.

Stage B of the pipeline. A `Representation` encodes a set of rendered pages into
a `Payload` for the reasoner. The four composers are the primary modality ladder
from the spec:

- `T`   -> raw text only.
- `TL`  -> text + layout/structure (strings only).
- `TLV` -> text + layout + visual (strings + page images).
- `V`   -> visual only (page images).

Two invariants are frozen here. First, the composer calls modular *channel*
functions (`tools/text.py`, `tools/layout.py`, `tools/visual.py`) rather than
building channels itself, so Stages 4-5 swap real tools in without touching this
file. Second, the modality boundary is structural: only `TLV` and `V` add image
parts; `T`/`TL` add strings. `Payload.__post_init__` re-checks this so a future
bug cannot leak an image into a text-only condition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from schema import Modality, Page, Part, Payload, TextPart
from tools.layout import layout_channel
from tools.text import text_channel
from tools.visual import visual_channel


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


class TextRepresentation(Representation):
    """`T`: raw text only."""

    modality: Modality = "T"

    def build(self, pages: Sequence[Page]) -> Payload:
        parts = _text_block("text", text_channel(pages))
        return Payload("T", tuple(parts))


class TextLayoutRepresentation(Representation):
    """`TL`: text + layout/structure, strings only."""

    modality: Modality = "TL"

    def build(self, pages: Sequence[Page]) -> Payload:
        parts = _text_block("text", text_channel(pages))
        parts += _text_block("layout", layout_channel(pages))
        return Payload("TL", tuple(parts))


class TextLayoutVisualRepresentation(Representation):
    """`TLV`: text + layout strings plus page images."""

    modality: Modality = "TLV"

    def build(self, pages: Sequence[Page]) -> Payload:
        parts: list[Part] = []
        parts += _text_block("text", text_channel(pages))
        parts += _text_block("layout", layout_channel(pages))
        parts += list(visual_channel(pages))
        return Payload("TLV", tuple(parts))


class VisualRepresentation(Representation):
    """`V`: page images only."""

    modality: Modality = "V"

    def build(self, pages: Sequence[Page]) -> Payload:
        return Payload("V", tuple(visual_channel(pages)))


#: Registry of the four ladder rungs, keyed by their modality name.
REPRESENTATIONS: dict[Modality, type[Representation]] = {
    "T": TextRepresentation,
    "TL": TextLayoutRepresentation,
    "TLV": TextLayoutVisualRepresentation,
    "V": VisualRepresentation,
}


def get_representation(modality: Modality) -> Representation:
    """Return a representation composer instance for a ladder rung."""

    try:
        return REPRESENTATIONS[modality]()
    except KeyError as exc:
        raise KeyError(f"unknown representation {modality!r}") from exc
