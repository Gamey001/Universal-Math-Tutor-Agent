"""Image loading and a cheap pre-check for photos with nothing on them.

A blank page, a lens-cap shot, or a completely smeared photo is flat
everywhere: once camera grain is smoothed away, no part of it has any
contrast. Catching that here means the learner gets a kind "I couldn't see
anything" right away instead of waiting on two vision models that would
only guess. Contrast is measured per tile, so one small equation on a big
white page still counts. Anything less obvious (blurry but not flat) is
left to the vision models.
"""

from __future__ import annotations

import io

from PIL import Image, ImageFilter, ImageOps, ImageStat

from .config import MAX_IMAGE_SIDE

# If no tile has more grayscale contrast than this (after denoising), the image is empty.
# Calibrated: white/grainy/unevenly lit blank paper scores < 2, any visible marks > 5.
BLANK_TILE_STDDEV = 4.0
TILE_GRID = 8


def load_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)  # phone photos are often stored sideways
    return img.convert("RGB")


def prepare_for_model(data: bytes) -> bytes:
    """Downscale to a size vision models handle quickly, as PNG bytes."""
    img = load_image(data)
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def detect_image_issue(data: bytes) -> str | None:
    """Return "unopenable", "blank", or None if the image is worth reading."""
    try:
        img = load_image(data)
    except Exception:
        return "unopenable"
    gray = img.convert("L")
    gray.thumbnail((512, 512))
    gray = gray.filter(ImageFilter.GaussianBlur(2))  # smooth away camera grain
    w, h = gray.size
    tiles = (
        gray.crop((i * w // TILE_GRID, j * h // TILE_GRID, (i + 1) * w // TILE_GRID, (j + 1) * h // TILE_GRID))
        for i in range(TILE_GRID) for j in range(TILE_GRID)
    )
    if max(ImageStat.Stat(t).stddev[0] for t in tiles) < BLANK_TILE_STDDEV:
        return "blank"
    return None
