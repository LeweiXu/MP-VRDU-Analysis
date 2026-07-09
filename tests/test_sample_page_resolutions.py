"""Tests for resolution-sample image sizing."""

from __future__ import annotations

from PIL import Image

from ops.scripts.sample_page_resolutions import (
    dimensions_for_pixel_cap,
    resize_for_pixel_cap,
)


def test_dimensions_fit_cap_and_patch_grid() -> None:
    width, height = dimensions_for_pixel_cap(1200, 1800, 250_880)

    assert width * height <= 250_880
    assert width % 28 == 0
    assert height % 28 == 0
    assert abs((width / height) - (1200 / 1800)) < 0.05


def test_resize_does_not_upscale_small_images() -> None:
    image = Image.new("RGB", (280, 420))

    resized = resize_for_pixel_cap(image, 1_003_520)

    assert resized.size == image.size
