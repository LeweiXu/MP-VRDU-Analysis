"""Page-image channel builder and per-page vision-token estimation from the
resolution preset."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from schema import ImagePart, Page


@dataclass(frozen=True)
class VisualArtifact:
    """One page image with cost/provenance metadata."""

    part: ImagePart
    page_index: int
    width: int
    height: int
    token_cost_estimate: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _image_size(path: Path) -> tuple[int, int]:
    """Return image dimensions using Pillow."""

    from PIL import Image

    with Image.open(path) as image:
        return image.size


def estimate_visual_tokens(width: int, height: int, patch: int = 28) -> int:
    """Return a rough patch-count vision token estimate for one image."""

    return max(1, math.ceil(width / patch) * math.ceil(height / patch))


def tokens_for_pixel_cap(max_pixels: int, patch: int = 28) -> int:
    """Vision tokens a square image capped at `max_pixels` produces (one page)."""

    side = math.sqrt(max(1, max_pixels))
    return max(1, math.ceil(side / patch) ** 2)


def _artifact(page: Page, image_path: Path, *, source: str, metadata: dict[str, Any] | None = None) -> VisualArtifact:
    """Build one visual artifact from an image path."""

    width, height = _image_size(image_path)
    return VisualArtifact(
        part=ImagePart(image_path=image_path),
        page_index=page.index,
        width=width,
        height=height,
        token_cost_estimate=estimate_visual_tokens(width, height),
        source=source,
        metadata=metadata or {},
    )


def full_page(pages: Sequence[Page]) -> tuple[VisualArtifact, ...]:
    """Return one full-page image artifact for every rendered page."""

    artifacts: list[VisualArtifact] = []
    for page in pages:
        if page.image_path is None:
            raise ValueError(f"page {page.index} has no image_path")
        artifacts.append(_artifact(page, page.image_path, source="full_page"))
    return tuple(artifacts)


def resolution(pages: Sequence[Page], scale: float) -> tuple[VisualArtifact, ...]:
    """Return page images rescaled by `scale`, with updated token estimates.

    Used by the resolution sweep to characterise sensitivity; the fixed
    deployment preset is applied at the reasoner image processor via its
    per-page pixel cap, so a normal cell renders at scale 1.
    """

    if scale <= 0:
        raise ValueError("scale must be positive")
    if scale == 1:
        return full_page(pages)

    from PIL import Image

    artifacts: list[VisualArtifact] = []
    for page in pages:
        if page.image_path is None:
            raise ValueError(f"page {page.index} has no image_path")
        source_path = Path(page.image_path)
        target_path = source_path.with_name(f"{source_path.stem}__scale{scale:g}{source_path.suffix}")
        if not target_path.exists():
            with Image.open(source_path) as image:
                width = max(1, int(round(image.width * scale)))
                height = max(1, int(round(image.height * scale)))
                resized = image.resize((width, height), Image.Resampling.LANCZOS)
                resized.save(target_path)
        artifacts.append(
            _artifact(
                page,
                target_path,
                source="resolution",
                metadata={"scale": scale, "source_image": str(source_path)},
            )
        )
    return tuple(artifacts)


def visual_channel(pages: Sequence[Page]) -> tuple[ImagePart, ...]:
    """Return one image part per page that has a rendered image."""

    return tuple(artifact.part for artifact in full_page(pages))
