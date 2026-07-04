"""Visual page-image, crop, and resolution tools for image-bearing representations.

Stage 3 placeholder: `visual_channel` returns one `ImagePart` per rendered page
image. Stage M2 (v3) replaces this with the ladder's image variants (`full_page`
and `resolution`; `region_crop` degrades to page-level per the Stage-1 verdict),
each carrying a token-cost estimate, behind the same return type so the
`TLV`/`V` composers keep calling `visual_channel`.
"""

from __future__ import annotations

from collections.abc import Sequence

from schema import ImagePart, Page


def visual_channel(pages: Sequence[Page]) -> tuple[ImagePart, ...]:
    """Return one image part per page that has a rendered image."""

    return tuple(
        ImagePart(image_path=page.image_path)
        for page in pages
        if page.image_path is not None
    )
