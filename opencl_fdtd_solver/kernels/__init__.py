# Copyright (C) 2026: OpenCL FDTD Solver Contributors
#
# This file is part of opencl-fdtd-solver.

"""OpenCL C and CUDA C kernel sources for the FDTD engines."""

from __future__ import annotations

from pathlib import Path

_KERNEL_DIR = Path(__file__).resolve().parent

# Compile-time concatenation order (must match historical monolithic source).
KERNEL_FILES = (
    "yee_update.cl",
    "sources.cl",
    "dft_farfield.cl",
)

# CUDA ports of the same kernels, templated on a ``real`` typedef.
CUDA_KERNEL_FILES = (
    "yee_update.cu",
    "sources.cu",
    "dft_farfield.cu",
)


def load_kernel_source(names: tuple[str, ...] = KERNEL_FILES) -> str:
    """Load and concatenate packaged ``.cl`` files into one OpenCL program source."""
    chunks = []
    for name in names:
        path = _KERNEL_DIR / name
        chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def load_cuda_kernel_source(names: tuple[str, ...] = CUDA_KERNEL_FILES) -> str:
    """Load and concatenate packaged ``.cu`` files into one CUDA program source."""
    return load_kernel_source(names)
