# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
#
# This file is part of opencl-fdtd-solver.

"""Font-free OpenCL vs Meep far-field |S|(θ) overlay PNGs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .plane_render import read_png_rgb  # re-export for tests


def _peak_align_db(db: np.ndarray) -> np.ndarray:
    a = np.asarray(db, dtype=np.float64)
    return a - float(np.max(a))


def _draw_polyline(rgb: np.ndarray, xs: np.ndarray, ys: np.ndarray, color: tuple[int, int, int]) -> None:
    """Bresenham-style segments between consecutive samples (inclusive)."""
    h, w, _ = rgb.shape
    for i in range(len(xs) - 1):
        x0, y0 = int(xs[i]), int(ys[i])
        x1, y1 = int(xs[i + 1]), int(ys[i + 1])
        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        for t in range(n + 1):
            x = int(round(x0 + (x1 - x0) * t / n))
            y = int(round(y0 + (y1 - y0) * t / n))
            if 0 <= x < w and 0 <= y < h:
                rgb[y, x] = color
                # 1px neighbors for visibility
                for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < w and 0 <= yy < h:
                        rgb[yy, xx] = color


def compose_pattern_overlay(
    angles_deg: np.ndarray,
    ocl_db: np.ndarray,
    meep_db: np.ndarray,
    *,
    width: int = 380,
    height: int = 120,
    db_floor: float = -40.0,
) -> np.ndarray:
    """Return HxWx3 uint8: peak-aligned OpenCL (blue) over Meep (red) vs angle."""
    a = _peak_align_db(ocl_db)
    b = _peak_align_db(meep_db)
    n = len(a)
    if n < 2:
        raise ValueError("need at least 2 angles")
    rgb = np.full((height, width, 3), 255, dtype=np.uint8)
    # Mid gray baseline at 0 dB and floor.
    for y_frac in (0.0, 1.0):
        y = int(round(y_frac * (height - 1)))
        rgb[y, :] = (220, 220, 220)

    def db_to_y(db: np.ndarray) -> np.ndarray:
        t = (0.0 - db) / (0.0 - db_floor)
        t = np.clip(t, 0.0, 1.0)
        return (t * (height - 1)).astype(np.float64)

    xs = np.linspace(0, width - 1, n)
    _draw_polyline(rgb, xs, db_to_y(b), (200, 40, 40))  # Meep red underneath
    _draw_polyline(rgb, xs, db_to_y(a), (30, 90, 200))  # OpenCL blue on top
    return rgb


def write_pattern_overlay_png(
    path: str | Path,
    angles_deg: np.ndarray,
    ocl_db: np.ndarray,
    meep_db: np.ndarray,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = compose_pattern_overlay(angles_deg, ocl_db, meep_db)
    try:
        from PIL import Image

        Image.fromarray(rgb, mode="RGB").save(
            path, format="PNG", optimize=False, compress_level=6
        )
    except ImportError:
        from matplotlib.image import imsave

        imsave(path, rgb)
    return path


__all__ = [
    "compose_pattern_overlay",
    "write_pattern_overlay_png",
    "read_png_rgb",
]
