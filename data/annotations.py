"""Reads and validates the human document-label table (bin_label, scan_label,
dominant_visual)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from config import ROOT

# Default location of the hand-labelled document table. One row per document.
ANNOTATION_SHEET = ROOT / "annotations" / "doc_labels.csv"

BIN_LABELS = ("text-dominant", "mixed-modality", "visual-dominant")
SCAN_LABELS = ("digital", "scanned")
# dominant_visual is exploratory and multi-valued; these are the allowed tokens.
VISUAL_KINDS = ("tables", "charts", "figures", "photos", "none")


@dataclass(frozen=True)
class DocLabel:
    """One document's manual labels."""

    doc_id: str
    bin_label: str
    scan_label: str
    dominant_visual: tuple[str, ...] = ()


def _split_multi(value: str) -> tuple[str, ...]:
    """Split a `dominant_visual` cell (comma/semicolon/pipe separated)."""

    parts = [p.strip() for p in value.replace(";", ",").replace("|", ",").split(",")]
    return tuple(p for p in parts if p)


def validate_label(row: DocLabel) -> None:
    """Raise if any label in a row is outside its allowed set."""

    if row.bin_label not in BIN_LABELS:
        raise ValueError(f"{row.doc_id}: bin_label {row.bin_label!r} not in {BIN_LABELS}")
    if row.scan_label not in SCAN_LABELS:
        raise ValueError(f"{row.doc_id}: scan_label {row.scan_label!r} not in {SCAN_LABELS}")
    for kind in row.dominant_visual:
        if kind not in VISUAL_KINDS:
            raise ValueError(f"{row.doc_id}: dominant_visual {kind!r} not in {VISUAL_KINDS}")


REQUIRED_COLUMNS = ("doc_id", "bin_label", "scan_label")


@lru_cache(maxsize=8)
def load_annotations(path: str | None = None) -> Mapping[str, DocLabel]:
    """Load the label table into a `doc_id -> DocLabel` map, validating each row.

    Returns an empty map only when the sheet does not exist yet (the labelling
    pass has not run), so callers can degrade to blank labels. Once the sheet
    exists it is treated as authoritative: a missing required column or an
    out-of-set label raises, so a malformed sheet fails loudly rather than
    silently producing blank bins.
    """

    sheet = Path(path or ANNOTATION_SHEET)
    if not sheet.is_file():
        return {}
    labels: dict[str, DocLabel] = {}
    with sheet.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in fields]
        if missing_cols:
            raise ValueError(f"annotation sheet {sheet} is missing required columns: {missing_cols}")
        for raw in reader:
            doc_id = (raw.get("doc_id") or "").strip()
            if not doc_id:
                continue
            label = DocLabel(
                doc_id=doc_id,
                bin_label=(raw.get("bin_label") or "").strip(),
                scan_label=(raw.get("scan_label") or "").strip(),
                dominant_visual=_split_multi(raw.get("dominant_visual") or ""),
            )
            validate_label(label)
            labels[doc_id] = label
    return labels
