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

import warnings

import numpy as np

from ._validation import sheet_current, source_sheet, validate_grid
from .constants import C0, EPS0, MU0
from .cpml import build_cpml_profiles
from .materials import yee_edge_ce
from .plugin import SourceMonitorMixin


class NumPyFDTD(SourceMonitorMixin):
    """
    3D Yee-grid FDTD electromagnetic solver running entirely on CPU using NumPy.
    Acts as a reference implementation and fallback when OpenCL is unavailable.
    """

    def __init__(self, shape, dl, npml=20, dtype=np.float32, psi_dtype: np.dtype | None = None):
        shape, self.dl, self.npml = validate_grid(shape, dl, npml)
        self.Nx, self.Ny, self.Nz = shape
        self.dtype = dtype
        # Allow overriding CPML auxiliary-field dtype to reduce memory (e.g., np.float16).
        # Defaults to main computation dtype to preserve numerical parity.
        self.psi_dtype = self.dtype if psi_dtype is None else np.dtype(psi_dtype)
        self.t = 0.0
        self.step_num = 0

        # Courant-stable time step
        self.dt = 0.99 * self.dl / (C0 * np.sqrt(3.0))

        # Initialize Yee fields
        self.Ex = np.zeros(shape, dtype=dtype)
        self.Ey = np.zeros(shape, dtype=dtype)
        self.Ez = np.zeros(shape, dtype=dtype)
        self.Hx = np.zeros(shape, dtype=dtype)
        self.Hy = np.zeros(shape, dtype=dtype)
        self.Hz = np.zeros(shape, dtype=dtype)
        self.eps_r = np.ones(shape, dtype=dtype)
        self._ce_x = np.full(shape, self.dt / EPS0, dtype=dtype)
        self._ce_y = np.full(shape, self.dt / EPS0, dtype=dtype)
        self._ce_z = np.full(shape, self.dt / EPS0, dtype=dtype)

        self._sources = []
        self._monitors = []

        self._build_cpml()

    def set_epsilon(self, eps_array):
        expected = (self.Nx, self.Ny, self.Nz)
        if eps_array.shape != expected:
            raise ValueError(
                f"Epsilon shape mismatch: expected {expected}, got {tuple(eps_array.shape)}"
            )
        self.eps_r = eps_array.astype(self.dtype)
        self._ce_x, self._ce_y, self._ce_z = yee_edge_ce(self.eps_r, self.dt, dtype=self.dtype)

    def add_source_Ex(self, z_src, amp, i0=None, i1=None, j0=None, j1=None):
        """Soft-add a sheet amplitude directly onto ``Ex`` (legacy field inject).

        Prefer :meth:`add_source_Jx` when matching Meep current-density sources.

        Optional half-open index ranges ``[i0, i1)`` / ``[j0, j1)`` limit the
        sheet (default: full XY, including PML). Use interior-only bounds when
        matching Meep sources that stop at the PML.
        """
        warnings.warn(
            "add_source_Ex is a legacy Ex soft-add; prefer add_source_Jx for SI current density",
            DeprecationWarning,
            stacklevel=2,
        )
        z, i0_i, i1_i, j0_i, j1_i = source_sheet((self.Nx, self.Ny, self.Nz), z_src, i0, i1, j0, j1)
        self.Ex[i0_i:i1_i, j0_i:j1_i, z] += np.dtype(self.dtype).type(amp)

    def add_source_Jx(
        self,
        z_src,
        Jx,
        i0=None,
        i1=None,
        j0=None,
        j1=None,
        *,
        rim_taper=False,
        rim_edge=0.8,
        rim_renorm=True,
    ):
        """Inject SI current density ``Jx`` (A/m²) on a constant-z Ex sheet.

        Applies ``Ex += -dt/(ε₀ εᵣ) Jx`` using the host ε array, matching
        Meep's ``D -= J·dt`` then ``E = χ⁻¹ D`` (with SI ε₀ restored) and the
        OpenCL kernel of the same name.

        Optional half-open ``[i0, i1)`` / ``[j0, j1)`` sheet bounds (default: full XY).

        If ``rim_taper`` is true, multiplies by sheet rim weights (edges ×
        ``rim_edge``, corners × ``rim_edge²``). Default ``rim_edge=0.8`` was
        tuned against Meep continuous volume-source restriction on the mid-plane
        cases. With ``rim_renorm`` (default true), ``Jx`` is scaled so ∑weights
        equals the hard cell count (preserves net ∫J).
        """
        z, i0_i, i1_i, j0_i, j1_i = source_sheet((self.Nx, self.Ny, self.Nz), z_src, i0, i1, j0, j1)
        jx, re = sheet_current(Jx, i1_i - i0_i, j1_i - j0_i, rim_taper, rim_edge, rim_renorm)

        sl_i = slice(i0_i, i1_i)
        sl_j = slice(j0_i, j1_i)
        soft = -self._ce_x[sl_i, sl_j, z] * jx
        if rim_taper:
            nx_s = max(0, i1_i - i0_i)
            ny_s = max(0, j1_i - j0_i)
            w = np.ones((nx_s, ny_s), dtype=self.dtype)
            # Match OpenCL: a cell on both edges is weighted once per axis.
            if nx_s >= 1:
                w[0, :] *= re
            if nx_s >= 2:
                w[-1, :] *= re
            if ny_s >= 1:
                w[:, 0] *= re
            if ny_s >= 2:
                w[:, -1] *= re
            soft = soft * w

        def inject():
            self.Ex[sl_i, sl_j, z] += soft.astype(self.dtype, copy=False)

        self._inject_current(inject)

    def _build_cpml(self):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        profiles = build_cpml_profiles(
            (Nx, Ny, Nz), npml=self.npml, dl=self.dl, dt=self.dt, dtype=self.dtype
        )

        def _bcast(prof, axis: int):
            shape = [1, 1, 1]
            shape[axis] = -1
            return (
                prof.b.reshape(shape),
                prof.c.reshape(shape),
                prof.kappa.reshape(shape),
            )

        self._bx_h, self._cx_h, self._kx_h = _bcast(profiles.h[0], 0)
        self._by_h, self._cy_h, self._ky_h = _bcast(profiles.h[1], 1)
        self._bz_h, self._cz_h, self._kz_h = _bcast(profiles.h[2], 2)
        self._bx_e, self._cx_e, self._kx_e = _bcast(profiles.e[0], 0)
        self._by_e, self._cy_e, self._ky_e = _bcast(profiles.e[1], 1)
        self._bz_e, self._cz_e, self._kz_e = _bcast(profiles.e[2], 2)

        # CPML auxiliary variables
        dt_aux = self.psi_dtype
        self._psi_Hx_y = np.zeros((Nx, Ny, Nz), dtype=dt_aux)
        self._psi_Hx_z = np.zeros((Nx, Ny, Nz), dtype=dt_aux)
        self._psi_Hy_x = np.zeros((Nx, Ny, Nz), dtype=dt_aux)
        self._psi_Hy_z = np.zeros((Nx, Ny, Nz), dtype=dt_aux)
        self._psi_Hz_x = np.zeros((Nx, Ny, Nz), dtype=dt_aux)
        self._psi_Hz_y = np.zeros((Nx, Ny, Nz), dtype=dt_aux)

        self._psi_Ex_y = np.zeros((Nx, Ny, Nz), dtype=dt_aux)
        self._psi_Ex_z = np.zeros((Nx, Ny, Nz), dtype=dt_aux)
        self._psi_Ey_x = np.zeros((Nx, Ny, Nz), dtype=dt_aux)
        self._psi_Ey_z = np.zeros((Nx, Ny, Nz), dtype=dt_aux)
        self._psi_Ez_x = np.zeros((Nx, Ny, Nz), dtype=dt_aux)
        self._psi_Ez_y = np.zeros((Nx, Ny, Nz), dtype=dt_aux)

    def _fwd(self, F, axis):
        d = np.zeros_like(F)
        if axis == 0:
            d[:-1, :, :] = F[1:, :, :] - F[:-1, :, :]
        elif axis == 1:
            d[:, :-1, :] = F[:, 1:, :] - F[:, :-1, :]
        else:
            d[:, :, :-1] = F[:, :, 1:] - F[:, :, :-1]
        return d

    def _bwd(self, F, axis):
        d = np.zeros_like(F)
        if axis == 0:
            d[1:, :, :] = F[1:, :, :] - F[:-1, :, :]
        elif axis == 1:
            d[:, 1:, :] = F[:, 1:, :] - F[:, :-1, :]
        else:
            d[:, :, 1:] = F[:, :, 1:] - F[:, :, :-1]
        return d

    def _update_H(self):
        dtm = self.dt / MU0
        Ex, Ey, Ez = self.Ex, self.Ey, self.Ez

        dEz_dy = self._fwd(Ez, 1)
        dEy_dz = self._fwd(Ey, 2)
        dEx_dz = self._fwd(Ex, 2)
        dEz_dx = self._fwd(Ez, 0)
        dEy_dx = self._fwd(Ey, 0)
        dEx_dy = self._fwd(Ex, 1)

        self._psi_Hx_y[...] = self._by_h * self._psi_Hx_y + self._cy_h * dEz_dy
        self._psi_Hx_z[...] = self._bz_h * self._psi_Hx_z + self._cz_h * dEy_dz
        self._psi_Hy_x[...] = self._bx_h * self._psi_Hy_x + self._cx_h * dEz_dx
        self._psi_Hy_z[...] = self._bz_h * self._psi_Hy_z + self._cz_h * dEx_dz
        self._psi_Hz_x[...] = self._bx_h * self._psi_Hz_x + self._cx_h * dEy_dx
        self._psi_Hz_y[...] = self._by_h * self._psi_Hz_y + self._cy_h * dEx_dy

        self.Hx -= dtm * (
            dEz_dy / (self._ky_h * self.dl)
            + self._psi_Hx_y
            - dEy_dz / (self._kz_h * self.dl)
            - self._psi_Hx_z
        )
        self.Hy -= dtm * (
            dEx_dz / (self._kz_h * self.dl)
            + self._psi_Hy_z
            - dEz_dx / (self._kx_h * self.dl)
            - self._psi_Hy_x
        )
        self.Hz -= dtm * (
            dEy_dx / (self._kx_h * self.dl)
            + self._psi_Hz_x
            - dEx_dy / (self._ky_h * self.dl)
            - self._psi_Hz_y
        )

    def _update_E(self):
        Hx, Hy, Hz = self.Hx, self.Hy, self.Hz

        dHz_dy = self._bwd(Hz, 1)
        dHy_dz = self._bwd(Hy, 2)
        dHx_dz = self._bwd(Hx, 2)
        dHz_dx = self._bwd(Hz, 0)
        dHy_dx = self._bwd(Hy, 0)
        dHx_dy = self._bwd(Hx, 1)

        self._psi_Ex_y[...] = self._by_e * self._psi_Ex_y + self._cy_e * dHz_dy
        self._psi_Ex_z[...] = self._bz_e * self._psi_Ex_z + self._cz_e * dHy_dz
        self._psi_Ey_x[...] = self._bx_e * self._psi_Ey_x + self._cx_e * dHz_dx
        self._psi_Ey_z[...] = self._bz_e * self._psi_Ey_z + self._cz_e * dHx_dz
        self._psi_Ez_x[...] = self._bx_e * self._psi_Ez_x + self._cx_e * dHy_dx
        self._psi_Ez_y[...] = self._by_e * self._psi_Ez_y + self._cy_e * dHx_dy

        self.Ex += self._ce_x * (
            dHz_dy / (self._ky_e * self.dl)
            + self._psi_Ex_y
            - dHy_dz / (self._kz_e * self.dl)
            - self._psi_Ex_z
        )
        self.Ey += self._ce_y * (
            dHx_dz / (self._kz_e * self.dl)
            + self._psi_Ey_z
            - dHz_dx / (self._kx_e * self.dl)
            - self._psi_Ey_x
        )
        self.Ez += self._ce_z * (
            dHy_dx / (self._kx_e * self.dl)
            + self._psi_Ez_x
            - dHx_dy / (self._ky_e * self.dl)
            - self._psi_Ez_y
        )

    def step(self):
        self._step_fields()
        self.t += self.dt
        self.step_num += 1
        for mon in self._monitors:
            mon(self)

    def run(self, n_steps, progress_every=0):
        for i in range(n_steps):
            self.step()
            if progress_every and i % progress_every == 0:
                print(f"  step {i}/{n_steps}  t={self.t:.3e} s", flush=True)
