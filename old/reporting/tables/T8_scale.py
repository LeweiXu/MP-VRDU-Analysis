"""Table 8: scale sanity (is the frontier stable across model size?).

Purpose:
    Reuses the family-replication builder over the Qwen3-VL size series and tags
    each row with its `scale_family`. Currently sourced from G1 only (the 32B
    scale task G4 is out of scope on our V100s), so `reporting.build` gates it off
    until G4 exists.

Arguments:
    None. Import-only; `build_table8_scale_sanity(rows, ...)` takes judged rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from config import DEFAULT_BINS
from pipeline.orchestrator import ResultRow

from .T3_family import build_table3_family_replication


def build_table8_scale_sanity(
    rows: Sequence[ResultRow],
    *,
    bins: Sequence[str] = DEFAULT_BINS,
    margin_points: float = 3.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Build Table 8: model-scale sanity skeleton."""

    table = build_table3_family_replication(
        rows,
        bins=bins,
        margin_points=margin_points,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    if "model_spec" in table:
        table.insert(0, "scale_family", table["model_spec"].map(lambda value: str(value).split("-", 1)[0]))
    return table
