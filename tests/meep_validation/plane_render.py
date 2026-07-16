# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
#
# This file is part of opencl-fdtd-solver.

"""Font-free mid-plane Ex triptych PNGs for golden-image regression."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _clim_shared(a: np.ndarray, b: np.ndarray, *, percentile: float = 99.5) -> float:
    stack = np.abs(np.concatenate([np.asarray(a).ravel(), np.asarray(b).ravel()]))
    if stack.size == 0:
        return 1.0
    clim = float(np.percentile(stack, percentile))
    return max(clim, 1e-30)


def field_to_rgb(field: np.ndarray, clim: float, *, cmap_name: str = "coolwarm") -> np.ndarray:
    """Map signed field to HxWx3 uint8 with a fixed diverging colormap (no text)."""
    from matplotlib import colormaps

    a = np.asarray(field, dtype=np.float64).T  # display z ↑, x → (imshow origin=lower style)
    t = np.clip(0.5 + 0.5 * (a / float(clim)), 0.0, 1.0)
    rgba = colormaps[cmap_name](t)
    return (np.asarray(rgba[..., :3], dtype=np.float64) * 255.0).astype(np.uint8)


def compose_triptych(
    ocl: np.ndarray,
    meep: np.ndarray,
    *,
    sep_px: int = 2,
    cmap_name: str = "coolwarm",
) -> np.ndarray:
    """Return HxWx3 uint8: OpenCL | Meep | residual (raw, shared clim on first two)."""
    a = np.asarray(ocl, dtype=np.float64)
    b = np.asarray(meep, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    clim = _clim_shared(a, b)
    residual = a - b
    rlim = max(float(np.percentile(np.abs(residual), 99.5)), 1e-30)
    left = field_to_rgb(a, clim, cmap_name=cmap_name)
    mid = field_to_rgb(b, clim, cmap_name=cmap_name)
    right = field_to_rgb(residual, rlim, cmap_name=cmap_name)
    h = left.shape[0]
    sep = np.full((h, sep_px, 3), 255, dtype=np.uint8)
    return np.concatenate([left, sep, mid, sep, right], axis=1)


def write_triptych_png(
    path: str | Path,
    ocl: np.ndarray,
    meep: np.ndarray,
    *,
    sep_px: int = 2,
    cmap_name: str = "coolwarm",
) -> Path:
    """Write a deterministic RGB PNG (no fonts / axes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = compose_triptych(ocl, meep, sep_px=sep_px, cmap_name=cmap_name)
    try:
        from PIL import Image

        Image.fromarray(rgb, mode="RGB").save(path, format="PNG", optimize=False, compress_level=6)
    except ImportError:
        from matplotlib.image import imsave

        imsave(path, rgb)
    return path


def read_png_rgb(path: str | Path) -> np.ndarray:
    """Load PNG as HxWx3 uint8."""
    path = Path(path)
    try:
        from PIL import Image

        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    except ImportError:
        import matplotlib.image as mpimg

        img = mpimg.imread(path)
        if img.dtype != np.uint8:
            img = (np.clip(img[..., :3], 0, 1) * 255.0).astype(np.uint8)
        else:
            img = img[..., :3]
        arr = img
    return arr
