"""Parser comparison: the parser-derived TL/TLV rungs, per bin.

Which parser produced the TL/TLV text is a per-run property (the comparison runs
one parser per run_tag), so a single results file yields one parser's block; the
build labels it from the run's config.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._common import Table
from .headline import ladder_by_bin


def build(rows: Sequence[Any], *, parser_label: str = "", margin_points: float = 3.0) -> Table:
    """Accuracy of the parser-fed rungs (TL, TLV) by bin, for one parser."""

    oracle = [r for r in rows if getattr(r, "condition", "") == "oracle"]
    note = f"parser = {parser_label}" if parser_label else ""
    return ladder_by_bin(
        oracle or rows,
        key="parser",
        title="Parser comparison: TL/TLV accuracy by bin",
        rungs=("TL", "TLV"),
        margin_points=margin_points,
        with_frontier=False,
        note=note,
    )
