"""Resolution sweep: image-bearing rungs (TLV, V) by bin, across resolution presets.

Resolution is a per-cell field now, so one results file can hold several presets;
this pivots the image rungs by preset so you can read accuracy vs resolution.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._common import Table, acc_cell, bin_of, group_by, ordered_bins

# Presets in ascending pixel budget, so the swept columns read cheap -> expensive.
RES_ORDER = ("low", "med", "high")


def build(rows: Sequence[Any], *, resolution_label: str = "", margin_points: float = 3.0) -> Table:
    """Accuracy of the image-bearing rungs (TLV, V) by bin and resolution preset."""

    oracle = [r for r in rows if getattr(r, "condition", "") == "oracle"]
    image_rows = [r for r in (oracle or rows) if getattr(r, "representation", "") in ("TLV", "V")]

    seen_res = {getattr(r, "visual_resolution", "") for r in image_rows}
    present_res = [res for res in RES_ORDER if res in seen_res]
    present_res += sorted(seen_res - set(present_res))  # any unexpected labels, last

    columns = ["bin", "rung", *present_res, "n"]
    table_rows: list[list[str]] = []
    by_bin = group_by(image_rows, bin_of)
    for b in ordered_bins(image_rows):
        for rung in ("TLV", "V"):
            group = [r for r in by_bin[b] if getattr(r, "representation", "") == rung]
            if not group:
                continue
            by_res = group_by(group, lambda r: getattr(r, "visual_resolution", ""))
            cells = [acc_cell(by_res.get(res, [])) for res in present_res]
            table_rows.append([b, rung, *cells, str(len(group))])

    return Table(
        key="resolution",
        title="Resolution sweep: TLV/V accuracy by bin and preset",
        columns=columns,
        rows=table_rows,
        note="",
    )
