"""Figures for the results section (Stage 7). Matplotlib is lazy-imported.

- top-k curves: accuracy/recall vs k.
- method-comparison bars with the no-retrieval FLOOR and oracle CEILING drawn in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def _plt():
    import matplotlib
    matplotlib.use("Agg")          # headless / no display needed
    import matplotlib.pyplot as plt
    return plt


def topk_curve(series: dict[str, list[tuple[int, float]]], ylabel: str,
               out_path: str | Path, title: str = "") -> Path:
    """series: {label -> [(k, value), ...]}. One line per label."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, points in series.items():
        points = sorted(points)
        ax.plot([k for k, _ in points], [v for _, v in points], marker="o", label=label)
    ax.set_xlabel("top-k")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def method_bars(values: dict[str, float], ylabel: str, out_path: str | Path,
                floor: Optional[float] = None, ceiling: Optional[float] = None,
                title: str = "") -> Path:
    """Bar chart of metric per method, with optional floor/ceiling reference lines."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = list(values)
    positions = range(len(methods))
    ax.bar(positions, [values[m] for m in methods])
    if floor is not None:
        ax.axhline(floor, ls="--", color="grey", label=f"no-retrieval floor ({floor:.3f})")
    if ceiling is not None:
        ax.axhline(ceiling, ls="--", color="green", label=f"oracle ceiling ({ceiling:.3f})")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if floor is not None or ceiling is not None:
        ax.legend()
    ax.set_xticks(list(positions))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
