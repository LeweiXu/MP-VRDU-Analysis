"""Compose rendered pages into text, layout, and visual model payloads.

Purpose:
    Defines Stage B of the pipeline. A `Representation` converts rendered
    `Page` objects into a `Payload` while enforcing the modality boundary used
    by the representation ladder.

Pipeline role:
    The orchestrator calls `get_representation(modality).build(pages)` after
    input conditioning and rendering. The four composers mirror the spec:

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

Arguments:
    None. This module is import-only; callers pass a modality name to
    `get_representation()`.
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


VALID_CHANNELS = ("T", "L", "V")
VALID_REPRESENTATIONS = {
    "".join(channel for i, channel in enumerate(VALID_CHANNELS) if mask & (1 << i))
    for mask in range(1, 1 << len(VALID_CHANNELS))
}


class ChannelRepresentation(Representation):
    """Any explicit combination of text/layout/vision channels.

    Valid names are ordered channel strings over `T`, `L`, and `V`: `T`, `L`,
    `V`, `TL`, `TV`, `LV`, and `TLV`. This preserves the original ladder while
    allowing YAML experiments to probe non-additive combinations.
    """

    def __init__(self, modality: str) -> None:
        self.modality = modality

    def build(self, pages: Sequence[Page]) -> Payload:
        parts: list[Part] = []
        if "T" in self.modality:
            parts += _text_block("text", text_channel(pages))
        if "L" in self.modality:
            parts += _text_block("layout", layout_channel(pages))
        if "V" in self.modality:
            parts += list(visual_channel(pages))
        return Payload(self.modality, tuple(parts))


def get_representation(modality: Modality) -> Representation:
    """Return a representation composer instance for a ladder rung."""

    name = str(modality)
    if name not in VALID_REPRESENTATIONS:
        raise KeyError(
            f"unknown representation {name!r}; use an ordered non-empty "
            "combination of T, L, V (T, L, V, TL, TV, LV, TLV)"
        )
    return ChannelRepresentation(name)
