"""Mined: peak-VRAM headroom per spec/rung/resolution against the 16 GB V100 ceiling.

Answers "how close does each config run to the memory ceiling", which predicts what
fits on accessible hardware. VRAM is a clean cost signal (unaffected by the
verbose-answer change).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._common import Table, group_by

# The V100 memory ceiling the deployment story is framed against.
V100_CEILING_MB = 16384.0


def _percentile_mb(values: Sequence[float], q: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, int(round(q * (len(vals) - 1))))
    return vals[idx] / 1e6


def build(rows: Sequence[Any]) -> Table:
    """(model_spec, rung, resolution) -> peak VRAM (max + p95) and headroom to 16 GB."""

    usable = [r for r in rows if getattr(r, "peak_vram_bytes", 0)]
    columns = ["model_spec", "rung", "resolution", "peak_vram_mb_max", "peak_vram_mb_p95", "headroom_mb", "n"]

    def key(r: Any) -> tuple[str, str, str]:
        return (getattr(r, "model_spec", ""), getattr(r, "representation", ""), getattr(r, "visual_resolution", ""))

    table_rows: list[list[str]] = []
    for (spec, rung, res), group in sorted(group_by(usable, key).items()):
        vram = [int(getattr(r, "peak_vram_bytes", 0)) for r in group]
        max_mb = max(vram) / 1e6
        p95_mb = _percentile_mb(vram, 0.95)
        headroom = V100_CEILING_MB - max_mb
        table_rows.append([
            spec, rung, res, f"{max_mb:.0f}", f"{p95_mb:.0f}", f"{headroom:.0f}", str(len(group)),
        ])
    return Table(
        key="mined_vram_headroom",
        title="Mined: peak-VRAM headroom vs the 16 GB V100 ceiling",
        columns=columns,
        rows=table_rows,
        note="headroom_mb = 16384 - peak_vram_mb_max; a negative value means the config OOMs a V100.",
    )


def summary(rows: Sequence[Any]) -> Table:
    """Worst-case peak VRAM + headroom per model_spec, pooled over all rungs/resolutions."""

    usable = [r for r in rows if getattr(r, "peak_vram_bytes", 0)]
    columns = ["model_spec", "peak_vram_mb_max", "headroom_mb", "n"]
    table_rows: list[list[str]] = []
    for spec, group in sorted(group_by(usable, lambda r: getattr(r, "model_spec", "")).items()):
        max_mb = max(int(getattr(r, "peak_vram_bytes", 0)) for r in group) / 1e6
        table_rows.append([spec, f"{max_mb:.0f}", f"{V100_CEILING_MB - max_mb:.0f}", str(len(group))])
    return Table(key="vram_headroom_summary", title="VRAM headroom (overall): worst-case peak per model_spec",
                 columns=columns, rows=table_rows,
                 note="peak = max over all rungs/resolutions; negative headroom means the spec OOMs a V100 at its heaviest config.")
