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

"""Yee-lattice material sampling helpers."""

from __future__ import annotations

import numpy as np

from .constants import EPS0


def yee_edge_eps(eps_r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arithmetic averages of cell-centered ``εᵣ`` onto Yee E-edge locations.

    ``Ex`` lives at ``(i+½, j, k)``, ``Ey`` at ``(i, j+½, k)``, ``Ez`` at
    ``(i, j, k+½)``. Each component uses the mean of the two cells that share
    that edge (the last plane along the stagger axis keeps the cell value).
    """
    eps = np.asarray(eps_r, dtype=np.float64)
    eps_x = eps.copy()
    eps_x[:-1] = 0.5 * (eps[:-1] + eps[1:])
    eps_y = eps.copy()
    eps_y[:, :-1] = 0.5 * (eps[:, :-1] + eps[:, 1:])
    eps_z = eps.copy()
    eps_z[:, :, :-1] = 0.5 * (eps[:, :, :-1] + eps[:, :, 1:])
    return eps_x, eps_y, eps_z


def yee_edge_ce(
    eps_r: np.ndarray, dt: float, dtype=np.float32
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-component ``ce = dt/(ε₀ εᵣ)`` at Yee E-edge samples."""
    eps_x, eps_y, eps_z = yee_edge_eps(eps_r)
    scale = float(dt) / EPS0
    dt_t = np.dtype(dtype)
    return (
        (scale / eps_x).astype(dt_t),
        (scale / eps_y).astype(dt_t),
        (scale / eps_z).astype(dt_t),
    )
