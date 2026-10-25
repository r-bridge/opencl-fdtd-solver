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


def _yee_edge_average(values: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    """Average cell-centered values over cells adjacent to an edge."""
    total = np.zeros(values.shape, dtype=np.float64)
    count = np.zeros(values.shape, dtype=np.uint8)
    for offset_a in (0, -1):
        for offset_b in (0, -1):
            dst = [slice(None)] * 3
            src = [slice(None)] * 3
            for axis, offset in ((axes[0], offset_a), (axes[1], offset_b)):
                if offset == -1:
                    dst[axis] = slice(1, None)
                    src[axis] = slice(None, -1)
            total[tuple(dst)] += values[tuple(src)]
            count[tuple(dst)] += 1
    np.divide(total, count, out=total)
    return total


def yee_edge_eps(eps_r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average cell-centered ``εᵣ`` onto Yee E-edge locations.

    ``Ex`` lives at ``(i+½, j, k)``, ``Ey`` at ``(i, j+½, k)``, ``Ez`` at
    ``(i, j, k+½)``. Each edge averages the cells adjacent to it in the two
    transverse directions (four cells in the interior); at domain boundaries,
    only in-domain neighboring cells contribute.
    """
    eps = np.asarray(eps_r, dtype=np.float64)
    return (
        _yee_edge_average(eps, (1, 2)),
        _yee_edge_average(eps, (0, 2)),
        _yee_edge_average(eps, (0, 1)),
    )


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


def yee_edge_pec(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell-centered PEC mask → per-component Yee E-edge masks.

    Cell ``(i, j, k)`` spans ``[i, i+1] × [j, j+1] × [k, k+1]``. ``Ex[i, j, k]``
    lies at ``(i+½, j, k)``: it runs along x across cell ``i`` and sits on the
    node line ``(j, k)``, so it is one of the four x-directed boundary edges of
    the cells ``(i, j-1..j, k-1..k)``. The Ex mask is therefore the OR of the
    cell mask over its two *transverse* neighbours (``j-1`` and ``k-1``),
    clipped at the domain boundary; likewise Ey over ``i-1``/``k-1`` and Ez
    over ``i-1``/``j-1``. Every one of the 12 edges of a masked cell ends up
    masked, so finite and staircased conductors present a closed surface with
    no active tangential edge. (Expanding along the component's *own* axis
    instead would clamp collinear edges outside the cell and leave the
    enclosing transverse edges free.)

    Used by ``OpenCLFDTD.set_epsilon(pec_mask=)`` to pin those edges to
    E = 0 (``ca = ce = 0``), the correct σ→∞ limit — feeding a huge σ into
    ``yee_edge_ca_cb`` instead drives ``Ca → −1`` (sign-flipping, stable but
    wrong), and a huge lossless εᵣ demands resolving the internal
    wavelength. PEC needs neither: foil-like reflectors get exact tangential
    E = 0 at any grid resolution.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 3:
        raise ValueError(f"PEC mask must be 3-dimensional, got {m.ndim}D")

    def _or_prev(a: np.ndarray, axis: int) -> np.ndarray:
        """OR each cell with its predecessor along ``axis`` (index 0 keeps itself)."""
        out = a.copy()
        dst = [slice(None)] * 3
        src = [slice(None)] * 3
        dst[axis] = slice(1, None)
        src[axis] = slice(None, -1)
        out[tuple(dst)] |= a[tuple(src)]
        return out

    mx = _or_prev(_or_prev(m, 1), 2)
    my = _or_prev(_or_prev(m, 0), 2)
    mz = _or_prev(_or_prev(m, 0), 1)
    return mx, my, mz


def yee_edge_ca_cb(
    eps_r: np.ndarray, sigma: np.ndarray, dt: float, dtype=np.float32
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Per-component lossy-medium E-update coefficients at Yee edges, standard
    semi-implicit (trapezoidal) loss term (Taflove & Hagness):

        E^{n+1} = Ca*E^n + Cb*(curl(H) - J)
        Ca = (1 - sigma*dt/(2*eps0*eps_r)) / (1 + sigma*dt/(2*eps0*eps_r))
        Cb = (dt/(eps0*eps_r)) / (1 + sigma*dt/(2*eps0*eps_r))

    ``sigma=0`` everywhere recovers ``Ca=1``, ``Cb=yee_edge_ce(eps_r, dt)``
    exactly (the old lossless update). Returns ``((ca_x, ca_y, ca_z), (cb_x,
    cb_y, cb_z))``.
    """
    eps_x, eps_y, eps_z = yee_edge_eps(eps_r)
    sig_x, sig_y, sig_z = yee_edge_eps(sigma)
    dt_f = float(dt)
    dt_t = np.dtype(dtype)

    def _ca_cb(eps_e: np.ndarray, sig_e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        loss = sig_e * dt_f / (2.0 * EPS0 * eps_e)
        denom = 1.0 + loss
        ca = (1.0 - loss) / denom
        cb = (dt_f / (EPS0 * eps_e)) / denom
        return ca.astype(dt_t), cb.astype(dt_t)

    ca_x, cb_x = _ca_cb(eps_x, sig_x)
    ca_y, cb_y = _ca_cb(eps_y, sig_y)
    ca_z, cb_z = _ca_cb(eps_z, sig_z)
    return (ca_x, ca_y, ca_z), (cb_x, cb_y, cb_z)
