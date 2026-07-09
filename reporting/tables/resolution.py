"""Resolution sweep: image-bearing rungs (TLV, V) by bin, at one preset.

The visual-resolution preset is a per-run property, so a single results file is
one preset's block; the build labels it from the run's config.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._common import Table
from .headline import ladder_by_bin


def build(rows: Sequence[Any], *, resolution_label: str = "", margin_points: float = 3.0) -> Table:
    """Accuracy of the image-bearing rungs (TLV, V) by bin, for one preset."""

    oracle = [r for r in rows if getattr(r, "condition", "") == "oracle"]
    note = f"resolution = {resolution_label}" if resolution_label else ""
    return ladder_by_bin(
        oracle or rows,
        key="resolution",
        title="Resolution sweep: TLV/V accuracy by bin",
        rungs=("TLV", "V"),
        margin_points=margin_points,
        with_frontier=False,
        note=note,
    )
