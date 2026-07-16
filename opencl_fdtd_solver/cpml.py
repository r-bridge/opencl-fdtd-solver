# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
#
# This file is part of opencl-fdtd-solver.
#
# opencl-fdtd-solver is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# opencl-fdtd-solver is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with opencl-fdtd-solver.  If not, see <http://www.gnu.org/licenses/>.

"""Complex-frequency-shifted PML (CPML) 1-D profile helpers."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .constants import EPS0, ETA0

# Polynomial grading order (Taflove).
CPML_M = 3
# Outer-boundary κ stretch (κ ≡ 1 was the previous default).
CPML_KAPPA_MAX = 15.0
# CFS α at the vacuum interface (decays to 0 at the outer wall).
CPML_ALPHA_MAX_OVER_ETA0 = 0.05


class CpmlAxisProfile(NamedTuple):
    """Per-axis CPML arrays ``b``, ``c``, ``κ`` sampled on the Yee grid."""

    b: np.ndarray
    c: np.ndarray
    kappa: np.ndarray


class CpmlProfiles(NamedTuple):
    """E-node and H-node CPML profiles for x/y/z (half-cell stagger)."""

    e: tuple[CpmlAxisProfile, CpmlAxisProfile, CpmlAxisProfile]
    h: tuple[CpmlAxisProfile, CpmlAxisProfile, CpmlAxisProfile]


def sigma_max(dl: float, *, m: int = CPML_M) -> float:
    """Taflove optimum ``σ_max ≈ 0.8(m+1)/(η₀ Δ)`` at the outer PML cell."""
    return 0.8 * (m + 1) / (ETA0 * float(dl))


def _xi(npml: int, i: int, *, lo: bool, node_offset: float) -> float:
    """Normalised depth into the PML (1 at outer wall, ~0 at vacuum interface).

    ``node_offset`` is 0 for E-aligned samples and 0.5 for H-aligned samples
    (half-cell stagger between the dual Yee grids).
    """
    if lo:
        return (npml - i - node_offset) / npml
    return (i + 1 - node_offset) / npml


def build_cpml_1d(
    n: int,
    *,
    npml: int,
    dl: float,
    dt: float,
    node_offset: float,
    dtype=np.float32,
    m: int | None = None,
    kappa_max: float | None = None,
    alpha_max: float | None = None,
) -> CpmlAxisProfile:
    """Build 1-D CPML ``(b, c, κ)`` of length ``n`` for one field stagger."""
    if m is None:
        m = CPML_M
    if kappa_max is None:
        kappa_max = CPML_KAPPA_MAX
    b = np.ones(n, dtype=dtype)
    c = np.zeros(n, dtype=dtype)
    kappa = np.ones(n, dtype=dtype)
    if npml <= 0 or n <= 0:
        return CpmlAxisProfile(b, c, kappa)

    sig_max = sigma_max(dl, m=m)
    if alpha_max is None:
        alpha_max = CPML_ALPHA_MAX_OVER_ETA0 / ETA0

    for i in range(npml):
        for lo, idx in ((True, i), (False, n - npml + i)):
            if idx < 0 or idx >= n:
                continue
            xi = max(_xi(npml, i, lo=lo, node_offset=node_offset), 0.0)
            sig = sig_max * xi**m
            kap = 1.0 + (kappa_max - 1.0) * xi**m
            alp = alpha_max * (1.0 - xi)
            decay = (sig / kap + alp) * dt / EPS0
            b[idx] = np.exp(-decay)
            denom = sig + kap * alp
            c[idx] = 0.0 if denom == 0 else sig / kap * (b[idx] - 1.0) / denom / dl
            kappa[idx] = kap
    return CpmlAxisProfile(b, c, kappa)


def build_cpml_profiles(
    shape: tuple[int, int, int],
    *,
    npml: int,
    dl: float,
    dt: float,
    dtype=np.float32,
    m: int | None = None,
    kappa_max: float | None = None,
) -> CpmlProfiles:
    """E-node (offset 0) and H-node (offset ½) profiles for all three axes."""
    nx, ny, nz = (int(v) for v in shape)
    kwargs = dict(npml=npml, dl=dl, dt=dt, dtype=dtype, m=m, kappa_max=kappa_max)
    e = (
        build_cpml_1d(nx, node_offset=0.0, **kwargs),
        build_cpml_1d(ny, node_offset=0.0, **kwargs),
        build_cpml_1d(nz, node_offset=0.0, **kwargs),
    )
    h = (
        build_cpml_1d(nx, node_offset=0.5, **kwargs),
        build_cpml_1d(ny, node_offset=0.5, **kwargs),
        build_cpml_1d(nz, node_offset=0.5, **kwargs),
    )
    return CpmlProfiles(e=e, h=h)
