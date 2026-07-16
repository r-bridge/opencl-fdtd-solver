# Copyright (C) 2026: OpenCL FDTD Solver Contributors
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

"""NumPy FDTD with face-local CPML ψ buffers (OpenCL memory layout)."""

from __future__ import annotations

import numpy as np

from .constants import MU0
from .cpml import build_cpml_profiles
from .numpy_engine import NumPyFDTD


class NumPyFDTD_FaceCPML(NumPyFDTD):
    """NumPy FDTD reference with face-striped CPML auxiliary fields.

    ψ storage matches :class:`~opencl_fdtd_solver.engine.OpenCLFDTD`:

    - x-normal: ``(2·npml, Ny, Nz)`` — ``Hy_x``, ``Hz_x``, ``Ey_x``, ``Ez_x``
    - y-normal: ``(Nx, 2·npml, Nz)`` — ``Hx_y``, ``Hz_y``, ``Ex_y``, ``Ez_y``
    - z-normal: ``(Nx, Ny, 2·npml)`` — ``Hx_z``, ``Hy_z``, ``Ex_z``, ``Ey_z``

    Numerically equivalent to volume :class:`NumPyFDTD` when ``c≡0`` outside
    the PML (ψ stays zero there). Prefer this class for large CPU grids.
    """

    CPML_STORAGE = "face"

    def _build_cpml(self):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        npml = self.npml
        profiles = build_cpml_profiles(
            (Nx, Ny, Nz), npml=npml, dl=self.dl, dt=self.dt, dtype=self.dtype
        )

        # 1-D CFS coeffs (E- and H-node staggers), matching OpenCL host buffers.
        self._bx_h = profiles.h[0].b
        self._cx_h = profiles.h[0].c
        self._kx_h = profiles.h[0].kappa
        self._by_h = profiles.h[1].b
        self._cy_h = profiles.h[1].c
        self._ky_h = profiles.h[1].kappa
        self._bz_h = profiles.h[2].b
        self._cz_h = profiles.h[2].c
        self._kz_h = profiles.h[2].kappa

        self._bx_e = profiles.e[0].b
        self._cx_e = profiles.e[0].c
        self._kx_e = profiles.e[0].kappa
        self._by_e = profiles.e[1].b
        self._cy_e = profiles.e[1].c
        self._ky_e = profiles.e[1].kappa
        self._bz_e = profiles.e[2].b
        self._cz_e = profiles.e[2].c
        self._kz_e = profiles.e[2].kappa

        # Broadcast κ for full-volume κ-scaled curls (same as volume NumPy).
        self._kx_h_b = self._kx_h.reshape(-1, 1, 1)
        self._ky_h_b = self._ky_h.reshape(1, -1, 1)
        self._kz_h_b = self._kz_h.reshape(1, 1, -1)
        self._kx_e_b = self._kx_e.reshape(-1, 1, 1)
        self._ky_e_b = self._ky_e.reshape(1, -1, 1)
        self._kz_e_b = self._kz_e.reshape(1, 1, -1)

        dt_aux = self.psi_dtype
        self.psi_x_size = (2 * npml * Ny * Nz) if npml > 0 else 0
        self.psi_y_size = (Nx * 2 * npml * Nz) if npml > 0 else 0
        self.psi_z_size = (Nx * Ny * 2 * npml) if npml > 0 else 0

        if npml <= 0:
            empty = (0,)
            self._psi_Hy_x = np.zeros(empty, dtype=dt_aux)
            self._psi_Hz_x = np.zeros(empty, dtype=dt_aux)
            self._psi_Ey_x = np.zeros(empty, dtype=dt_aux)
            self._psi_Ez_x = np.zeros(empty, dtype=dt_aux)
            self._psi_Hx_y = np.zeros(empty, dtype=dt_aux)
            self._psi_Hz_y = np.zeros(empty, dtype=dt_aux)
            self._psi_Ex_y = np.zeros(empty, dtype=dt_aux)
            self._psi_Ez_y = np.zeros(empty, dtype=dt_aux)
            self._psi_Hx_z = np.zeros(empty, dtype=dt_aux)
            self._psi_Hy_z = np.zeros(empty, dtype=dt_aux)
            self._psi_Ex_z = np.zeros(empty, dtype=dt_aux)
            self._psi_Ey_z = np.zeros(empty, dtype=dt_aux)
            return

        sx = (2 * npml, Ny, Nz)
        sy = (Nx, 2 * npml, Nz)
        sz = (Nx, Ny, 2 * npml)
        self._psi_Hy_x = np.zeros(sx, dtype=dt_aux)
        self._psi_Hz_x = np.zeros(sx, dtype=dt_aux)
        self._psi_Ey_x = np.zeros(sx, dtype=dt_aux)
        self._psi_Ez_x = np.zeros(sx, dtype=dt_aux)
        self._psi_Hx_y = np.zeros(sy, dtype=dt_aux)
        self._psi_Hz_y = np.zeros(sy, dtype=dt_aux)
        self._psi_Ex_y = np.zeros(sy, dtype=dt_aux)
        self._psi_Ez_y = np.zeros(sy, dtype=dt_aux)
        self._psi_Hx_z = np.zeros(sz, dtype=dt_aux)
        self._psi_Hy_z = np.zeros(sz, dtype=dt_aux)
        self._psi_Ex_z = np.zeros(sz, dtype=dt_aux)
        self._psi_Ey_z = np.zeros(sz, dtype=dt_aux)

    @staticmethod
    def _face_slices(n: int, npml: int):
        """Return (domain_slice, stripe_slice) for lo and hi PML faces."""
        return (
            (slice(0, npml), slice(0, npml)),
            (slice(n - npml, n), slice(npml, 2 * npml)),
        )

    def _update_H(self):
        dtm = self.dt / MU0
        dl = self.dl
        Ex, Ey, Ez = self.Ex, self.Ey, self.Ez
        npml = self.npml

        dEz_dy = self._fwd(Ez, 1)
        dEy_dz = self._fwd(Ey, 2)
        dEx_dz = self._fwd(Ex, 2)
        dEz_dx = self._fwd(Ez, 0)
        dEy_dx = self._fwd(Ey, 0)
        dEx_dy = self._fwd(Ex, 1)

        # κ-scaled curls on the full volume (κ≡1 outside PML).
        self.Hx -= dtm * (dEz_dy / (self._ky_h_b * dl) - dEy_dz / (self._kz_h_b * dl))
        self.Hy -= dtm * (dEx_dz / (self._kz_h_b * dl) - dEz_dx / (self._kx_h_b * dl))
        self.Hz -= dtm * (dEy_dx / (self._kx_h_b * dl) - dEx_dy / (self._ky_h_b * dl))

        if npml <= 0:
            return

        # x-normal faces → ψ_Hy_x, ψ_Hz_x  (Hy -= … − ψ_Hy_x; Hz -= … + ψ_Hz_x)
        for i_sl, s_sl in self._face_slices(self.Nx, npml):
            bx = self._bx_h[i_sl].reshape(-1, 1, 1)
            cx = self._cx_h[i_sl].reshape(-1, 1, 1)
            self._psi_Hy_x[s_sl, :, :] = bx * self._psi_Hy_x[s_sl, :, :] + cx * dEz_dx[i_sl, :, :]
            self._psi_Hz_x[s_sl, :, :] = bx * self._psi_Hz_x[s_sl, :, :] + cx * dEy_dx[i_sl, :, :]
            self.Hy[i_sl, :, :] -= dtm * (-self._psi_Hy_x[s_sl, :, :])
            self.Hz[i_sl, :, :] -= dtm * self._psi_Hz_x[s_sl, :, :]

        # y-normal faces → ψ_Hx_y, ψ_Hz_y
        for j_sl, s_sl in self._face_slices(self.Ny, npml):
            by = self._by_h[j_sl].reshape(1, -1, 1)
            cy = self._cy_h[j_sl].reshape(1, -1, 1)
            self._psi_Hx_y[:, s_sl, :] = by * self._psi_Hx_y[:, s_sl, :] + cy * dEz_dy[:, j_sl, :]
            self._psi_Hz_y[:, s_sl, :] = by * self._psi_Hz_y[:, s_sl, :] + cy * dEx_dy[:, j_sl, :]
            self.Hx[:, j_sl, :] -= dtm * self._psi_Hx_y[:, s_sl, :]
            self.Hz[:, j_sl, :] -= dtm * (-self._psi_Hz_y[:, s_sl, :])

        # z-normal faces → ψ_Hx_z, ψ_Hy_z
        for k_sl, s_sl in self._face_slices(self.Nz, npml):
            bz = self._bz_h[k_sl].reshape(1, 1, -1)
            cz = self._cz_h[k_sl].reshape(1, 1, -1)
            self._psi_Hx_z[:, :, s_sl] = bz * self._psi_Hx_z[:, :, s_sl] + cz * dEy_dz[:, :, k_sl]
            self._psi_Hy_z[:, :, s_sl] = bz * self._psi_Hy_z[:, :, s_sl] + cz * dEx_dz[:, :, k_sl]
            self.Hx[:, :, k_sl] -= dtm * (-self._psi_Hx_z[:, :, s_sl])
            self.Hy[:, :, k_sl] -= dtm * self._psi_Hy_z[:, :, s_sl]

    def _update_E(self):
        dl = self.dl
        Hx, Hy, Hz = self.Hx, self.Hy, self.Hz
        npml = self.npml

        dHz_dy = self._bwd(Hz, 1)
        dHy_dz = self._bwd(Hy, 2)
        dHx_dz = self._bwd(Hx, 2)
        dHz_dx = self._bwd(Hz, 0)
        dHy_dx = self._bwd(Hy, 0)
        dHx_dy = self._bwd(Hx, 1)

        self.Ex += self._ce_x * (dHz_dy / (self._ky_e_b * dl) - dHy_dz / (self._kz_e_b * dl))
        self.Ey += self._ce_y * (dHx_dz / (self._kz_e_b * dl) - dHz_dx / (self._kx_e_b * dl))
        self.Ez += self._ce_z * (dHy_dx / (self._kx_e_b * dl) - dHx_dy / (self._ky_e_b * dl))

        if npml <= 0:
            return

        # x-normal → ψ_Ey_x, ψ_Ez_x
        # Ey += ce * (… − ψ_Ey_x); Ez += ce * (… + ψ_Ez_x)
        for i_sl, s_sl in self._face_slices(self.Nx, npml):
            bx = self._bx_e[i_sl].reshape(-1, 1, 1)
            cx = self._cx_e[i_sl].reshape(-1, 1, 1)
            self._psi_Ey_x[s_sl, :, :] = bx * self._psi_Ey_x[s_sl, :, :] + cx * dHz_dx[i_sl, :, :]
            self._psi_Ez_x[s_sl, :, :] = bx * self._psi_Ez_x[s_sl, :, :] + cx * dHy_dx[i_sl, :, :]
            self.Ey[i_sl, :, :] += self._ce_y[i_sl, :, :] * (-self._psi_Ey_x[s_sl, :, :])
            self.Ez[i_sl, :, :] += self._ce_z[i_sl, :, :] * self._psi_Ez_x[s_sl, :, :]

        # y-normal → ψ_Ex_y, ψ_Ez_y
        for j_sl, s_sl in self._face_slices(self.Ny, npml):
            by = self._by_e[j_sl].reshape(1, -1, 1)
            cy = self._cy_e[j_sl].reshape(1, -1, 1)
            self._psi_Ex_y[:, s_sl, :] = by * self._psi_Ex_y[:, s_sl, :] + cy * dHz_dy[:, j_sl, :]
            self._psi_Ez_y[:, s_sl, :] = by * self._psi_Ez_y[:, s_sl, :] + cy * dHx_dy[:, j_sl, :]
            self.Ex[:, j_sl, :] += self._ce_x[:, j_sl, :] * self._psi_Ex_y[:, s_sl, :]
            self.Ez[:, j_sl, :] += self._ce_z[:, j_sl, :] * (-self._psi_Ez_y[:, s_sl, :])

        # z-normal → ψ_Ex_z, ψ_Ey_z
        for k_sl, s_sl in self._face_slices(self.Nz, npml):
            bz = self._bz_e[k_sl].reshape(1, 1, -1)
            cz = self._cz_e[k_sl].reshape(1, 1, -1)
            self._psi_Ex_z[:, :, s_sl] = bz * self._psi_Ex_z[:, :, s_sl] + cz * dHy_dz[:, :, k_sl]
            self._psi_Ey_z[:, :, s_sl] = bz * self._psi_Ey_z[:, :, s_sl] + cz * dHx_dz[:, :, k_sl]
            self.Ex[:, :, k_sl] += self._ce_x[:, :, k_sl] * (-self._psi_Ex_z[:, :, s_sl])
            self.Ey[:, :, k_sl] += self._ce_y[:, :, k_sl] * self._psi_Ey_z[:, :, s_sl]
