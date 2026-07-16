# Copyright (C) 2026: OpenCL FDTD Solver Contributors
#
# This file is part of opencl-fdtd-solver.

"""OpenCL C kernel sources for the FDTD engine."""

from __future__ import annotations

from pathlib import Path

_KERNEL_DIR = Path(__file__).resolve().parent

# Compile-time concatenation order (must match historical monolithic source).
KERNEL_FILES = (
    "yee_update.cl",
    "sources.cl",
    "dft_farfield.cl",
)


def load_kernel_source(names: tuple[str, ...] = KERNEL_FILES) -> str:
    """Load and concatenate packaged ``.cl`` files into one OpenCL program source."""
    chunks = []
    for name in names:
        path = _KERNEL_DIR / name
        chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)
