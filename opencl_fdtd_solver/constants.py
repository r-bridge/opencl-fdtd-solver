# Copyright (C) 2026: OpenCL FDTD Solver Contributors
#
# This file is part of opencl-fdtd-solver.

"""Shared SI electromagnetic constants."""

from __future__ import annotations

import numpy as np

C0 = 299_792_458.0
MU0 = 4e-7 * np.pi
EPS0 = 1.0 / (MU0 * C0**2)
ETA0 = np.sqrt(MU0 / EPS0)

__all__ = ["C0", "MU0", "EPS0", "ETA0"]
