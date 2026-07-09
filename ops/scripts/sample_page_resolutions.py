#!/usr/bin/env python3
"""Render random MMLongBench document pages at every configured resolution.

The sampler draws pages uniformly without replacement from the staged PDF
corpus. Each selected page gets one PNG per ``VISUAL_RESOLUTION_PRESETS`` entry,
plus a small Markdown manifest listing the source page and output dimensions.

Example:
    python -m ops.scripts.sample_page_resolutions --n 5 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
from pathlib import Path

from PIL import Image

from config import DEFAULT_PATHS, VISUAL_RESOLUTION_PRESETS

DEFAULT_OUTPUT = DEFAULT_PATHS.results_dir / "resolution_samples"
PDF_DIR = DEFAULT_PATHS.data_dir / "mmlongbench" / "documents"
PATCH_SIZE = 28


def safe_name(value: str) -> str:
    """Return a compact filesystem-safe name."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "document"


def dimensions_for_pixel_cap(
    width: int,
    height: int,
    max_pixels: int,
    *,
    factor: int = PATCH_SIZE,
) -> tuple[int, int]:
    """Fit dimensions under ``max_pixels``, aligned to the model patch size."""

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if max_pixels < factor * factor:
        raise ValueError(f"max_pixels must be at least {factor * factor}")

    scale = min(1.0, math.sqrt(max_pixels / (width * height)))
    target_width = max(factor, math.floor(width * scale / factor) * factor)
    target_height = max(factor, math.floor(height * scale / factor) * factor)
    while target_width * target_height > max_pixels:
        if target_width >= target_height and target_width > factor:
            target_width -= factor
        elif target_height > factor:
            target_height -= factor
        else:
            break
    return target_width, target_height


def resize_for_pixel_cap(image: Image.Image, max_pixels: int) -> Image.Image:
    """Return an aspect-preserving, patch-aligned image below the pixel cap."""

    dimensions = dimensions_for_pixel_cap(*image.size, max_pixels)
    if dimensions == image.size:
        return image.copy()
    return image.resize(dimensions, Image.Resampling.LANCZOS)


def corpus_pages(pdf_dir: Path) -> list[tuple[Path, int]]:
    """Enumerate every zero-based PDF page in the corpus."""

    import fitz

    pages = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        with fitz.open(pdf) as document:
            pages.extend((pdf, page_index) for page_index in range(document.page_count))
    return pages


def render_source_page(pdf: Path, page_index: int, dpi: int) -> Image.Image:
    """Render one PDF page to RGB at the source DPI."""

    import fitz

    with fitz.open(pdf) as document:
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def sample_pages(
    *,
    n: int,
    seed: int,
    dpi: int,
    pdf_dir: Path,
    output: Path,
) -> list[dict[str, str | int]]:
    """Sample, render, and save pages; return manifest records."""

    if n < 1:
        raise ValueError("--n must be at least 1")
    if dpi < 72:
        raise ValueError("--dpi must be at least 72")
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pages = corpus_pages(pdf_dir)
    if n > len(pages):
        raise ValueError(f"--n={n} exceeds the corpus page count ({len(pages)})")

    selected = random.Random(seed).sample(pages, n)
    run_dir = output / f"seed_{seed}_n_{n}"
    run_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str | int]] = []
    markdown = [
        "# Resolution samples",
        "",
        f"Random seed: `{seed}` · pages sampled: `{n}` · source render DPI: `{dpi}`",
        "",
    ]

    # Low-to-high ordering is easier to inspect visually.
    presets = sorted(VISUAL_RESOLUTION_PRESETS.items(), key=lambda item: item[1])
    for sample_number, (pdf, page_index) in enumerate(selected, 1):
        sample_dir = run_dir / (
            f"{sample_number:02d}__{safe_name(pdf.stem)}__page_{page_index + 1:04d}"
        )
        sample_dir.mkdir(parents=True, exist_ok=True)
        source = render_source_page(pdf, page_index, dpi)
        markdown += [
            f"## {sample_number}. `{pdf.name}`, page {page_index + 1}",
            "",
            "| Preset | Pixel cap | Dimensions | Image |",
            "|---|---:|---:|---|",
        ]
        for preset, pixel_cap in presets:
            rendered = resize_for_pixel_cap(source, pixel_cap)
            image_path = sample_dir / f"{preset}.png"
            rendered.save(image_path, format="PNG", optimize=True)
            relative_path = image_path.relative_to(run_dir)
            record = {
                "sample": sample_number,
                "doc_id": pdf.name,
                "page_number": page_index + 1,
                "preset": preset,
                "pixel_cap": pixel_cap,
                "width": rendered.width,
                "height": rendered.height,
                "image": str(relative_path),
            }
            records.append(record)
            markdown.append(
                f"| {preset} | {pixel_cap} | {rendered.width}×{rendered.height} "
                f"| [view]({relative_path.as_posix()}) |"
            )
        markdown.append("")

    with (run_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (run_dir / "README.md").write_text("\n".join(markdown) + "\n")
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5, help="number of random pages (default: 5)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument("--dpi", type=int, default=144, help="source PDF render DPI (default: 144)")
    parser.add_argument("--pdf-dir", type=Path, default=PDF_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = sample_pages(
        n=args.n,
        seed=args.seed,
        dpi=args.dpi,
        pdf_dir=args.pdf_dir,
        output=args.output,
    )
    run_dir = args.output / f"seed_{args.seed}_n_{args.n}"
    print(
        f"rendered {args.n} pages × {len(VISUAL_RESOLUTION_PRESETS)} presets "
        f"({len(records)} images): {run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
