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

import numpy as np

from .plugin import SourceMonitorMixin

C0 = 299_792_458.0
MU0 = 4e-7 * np.pi
EPS0 = 1.0 / (MU0 * C0**2)
ETA0 = np.sqrt(MU0 / EPS0)


class NumPyFDTD(SourceMonitorMixin):
    """
    3D Yee-grid FDTD electromagnetic solver running entirely on CPU using NumPy.
    Acts as a reference implementation and fallback when OpenCL is unavailable.
    """

    def __init__(self, shape, dl, npml=20, dtype=np.float32):
        self.Nx, self.Ny, self.Nz = shape
        self.dl = float(dl)
        self.npml = int(npml)
        self.dtype = dtype
        self.t = 0.0
        self.step_num = 0

        # Courant-stable time step
        self.dt = 0.99 * dl / (C0 * np.sqrt(3.0))

        # Initialize Yee fields
        self.Ex = np.zeros(shape, dtype=dtype)
        self.Ey = np.zeros(shape, dtype=dtype)
        self.Ez = np.zeros(shape, dtype=dtype)
        self.Hx = np.zeros(shape, dtype=dtype)
        self.Hy = np.zeros(shape, dtype=dtype)
        self.Hz = np.zeros(shape, dtype=dtype)
        self.eps_r = np.ones(shape, dtype=dtype)

        self._sources = []
        self._monitors = []

        self._build_cpml()

    def set_epsilon(self, eps_array):
        assert eps_array.shape == (self.Nx, self.Ny, self.Nz)
        self.eps_r = eps_array.astype(self.dtype)

    def add_source_Ex(self, z_src, amp, i0=None, i1=None, j0=None, j1=None):
        """Soft-add a sheet amplitude directly onto ``Ex`` (legacy field inject).

        Prefer :meth:`add_source_Jx` when matching Meep current-density sources.

        Optional half-open index ranges ``[i0, i1)`` / ``[j0, j1)`` limit the
        sheet (default: full XY, including PML). Use interior-only bounds when
        matching Meep sources that stop at the PML.
        """
        i0_i = 0 if i0 is None else int(i0)
        i1_i = self.Nx if i1 is None else int(i1)
        j0_i = 0 if j0 is None else int(j0)
        j1_i = self.Ny if j1 is None else int(j1)
        z = int(z_src)
        self.Ex[i0_i:i1_i, j0_i:j1_i, z] += self.dtype(amp)

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
        i0_i = 0 if i0 is None else int(i0)
        i1_i = self.Nx if i1 is None else int(i1)
        j0_i = 0 if j0 is None else int(j0)
        j1_i = self.Ny if j1 is None else int(j1)
        z = int(z_src)
        jx = float(Jx)
        re = float(rim_edge)
        if rim_taper and rim_renorm:
            nx_s = max(0, i1_i - i0_i)
            ny_s = max(0, j1_i - j0_i)
            if nx_s >= 2 and ny_s >= 2:
                ni, nj = nx_s - 2, ny_s - 2
                wsum = ni * nj + re * (2 * ni + 2 * nj) + (re * re) * 4
                jx *= (nx_s * ny_s) / wsum

        sl_i = slice(i0_i, i1_i)
        sl_j = slice(j0_i, j1_i)
        soft = -(self.dt / (EPS0 * self.eps_r[sl_i, sl_j, z])) * jx
        if rim_taper:
            nx_s = max(0, i1_i - i0_i)
            ny_s = max(0, j1_i - j0_i)
            w = np.ones((nx_s, ny_s), dtype=self.dtype)
            # Match OpenCL: both lo/hi edge checks fire on a 1-cell span (×rim²).
            if nx_s >= 1:
                w[0, :] *= re
                w[-1, :] *= re
            if ny_s >= 1:
                w[:, 0] *= re
                w[:, -1] *= re
            soft = soft * w
        self.Ex[sl_i, sl_j, z] += soft.astype(self.dtype, copy=False)

    def _build_cpml(self):
        dl = self.dl
        dt = self.dt
        npml = self.npml
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz

        m = 3
        sigma_opt = 0.8 * (m + 1) / (2.0 * ETA0 * dl * npml)
        alpha_max = 0.05 / ETA0

        def _1d_coeffs(n):
            b = np.ones(n, dtype=self.dtype)
            c = np.zeros(n, dtype=self.dtype)
            k = np.ones(n, dtype=self.dtype)
            for i in range(npml):
                for lo, idx in ((True, i), (False, n - npml + i)):
                    xi = (npml - i) / npml if lo else (i + 1) / npml
                    sig = sigma_opt * xi**m
                    kap = 1.0
                    alp = alpha_max * (1.0 - xi) ** 1
                    decay = (sig / kap + alp) * dt / EPS0
                    b[idx] = np.exp(-decay)
                    denom = sig + kap * alp
                    c[idx] = 0.0 if denom == 0 else sig / kap * (b[idx] - 1.0) / denom / dl
                    k[idx] = kap
            return b, c, k

        bx, cx, kx = _1d_coeffs(Nx)
        by, cy, ky = _1d_coeffs(Ny)
        bz, cz, kz = _1d_coeffs(Nz)

        # Reshape for broadcasting
        self._bx = bx.reshape(Nx, 1, 1)
        self._cx = cx.reshape(Nx, 1, 1)
        self._kx = kx.reshape(Nx, 1, 1)
        self._by = by.reshape(1, Ny, 1)
        self._cy = cy.reshape(1, Ny, 1)
        self._ky = ky.reshape(1, Ny, 1)
        self._bz = bz.reshape(1, 1, Nz)
        self._cz = cz.reshape(1, 1, Nz)
        self._kz = kz.reshape(1, 1, Nz)

        # CPML auxiliary variables
        self._psi_Hx_y = np.zeros((Nx, Ny, Nz), dtype=self.dtype)
        self._psi_Hx_z = np.zeros((Nx, Ny, Nz), dtype=self.dtype)
        self._psi_Hy_x = np.zeros((Nx, Ny, Nz), dtype=self.dtype)
        self._psi_Hy_z = np.zeros((Nx, Ny, Nz), dtype=self.dtype)
        self._psi_Hz_x = np.zeros((Nx, Ny, Nz), dtype=self.dtype)
        self._psi_Hz_y = np.zeros((Nx, Ny, Nz), dtype=self.dtype)

        self._psi_Ex_y = np.zeros((Nx, Ny, Nz), dtype=self.dtype)
        self._psi_Ex_z = np.zeros((Nx, Ny, Nz), dtype=self.dtype)
        self._psi_Ey_x = np.zeros((Nx, Ny, Nz), dtype=self.dtype)
        self._psi_Ey_z = np.zeros((Nx, Ny, Nz), dtype=self.dtype)
        self._psi_Ez_x = np.zeros((Nx, Ny, Nz), dtype=self.dtype)
        self._psi_Ez_y = np.zeros((Nx, Ny, Nz), dtype=self.dtype)

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

        self._psi_Hx_y = self._by * self._psi_Hx_y + self._cy * dEz_dy
        self._psi_Hx_z = self._bz * self._psi_Hx_z + self._cz * dEy_dz
        self._psi_Hy_x = self._bx * self._psi_Hy_x + self._cx * dEz_dx
        self._psi_Hy_z = self._bz * self._psi_Hy_z + self._cz * dEx_dz
        self._psi_Hz_x = self._bx * self._psi_Hz_x + self._cx * dEy_dx
        self._psi_Hz_y = self._by * self._psi_Hz_y + self._cy * dEx_dy

        self.Hx -= dtm * (
            dEz_dy / (self._ky * self.dl)
            + self._psi_Hx_y
            - dEy_dz / (self._kz * self.dl)
            - self._psi_Hx_z
        )
        self.Hy -= dtm * (
            dEx_dz / (self._kz * self.dl)
            + self._psi_Hy_z
            - dEz_dx / (self._kx * self.dl)
            - self._psi_Hy_x
        )
        self.Hz -= dtm * (
            dEy_dx / (self._kx * self.dl)
            + self._psi_Hz_x
            - dEx_dy / (self._ky * self.dl)
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

        self._psi_Ex_y = self._by * self._psi_Ex_y + self._cy * dHz_dy
        self._psi_Ex_z = self._bz * self._psi_Ex_z + self._cz * dHy_dz
        self._psi_Ey_x = self._bx * self._psi_Ey_x + self._cx * dHz_dx
        self._psi_Ey_z = self._bz * self._psi_Ey_z + self._cz * dHx_dz
        self._psi_Ez_x = self._bx * self._psi_Ez_x + self._cx * dHy_dx
        self._psi_Ez_y = self._by * self._psi_Ez_y + self._cy * dHx_dy

        coeff = self.dt / (EPS0 * self.eps_r)
        self.Ex += coeff * (
            dHz_dy / (self._ky * self.dl)
            + self._psi_Ex_y
            - dHy_dz / (self._kz * self.dl)
            - self._psi_Ex_z
        )
        self.Ey += coeff * (
            dHx_dz / (self._kz * self.dl)
            + self._psi_Ey_z
            - dHz_dx / (self._kx * self.dl)
            - self._psi_Ey_x
        )
        self.Ez += coeff * (
            dHy_dx / (self._kx * self.dl)
            + self._psi_Ez_x
            - dHx_dy / (self._ky * self.dl)
            - self._psi_Ez_y
        )

    def step(self):
        self._update_H()
        for src in self._sources:
            src(self)
        self._update_E()
        self.t += self.dt
        self.step_num += 1
        for mon in self._monitors:
            mon(self)

    def run(self, n_steps, progress_every=0):
        for i in range(n_steps):
            self.step()
            if progress_every and i % progress_every == 0:
                print(f"  step {i}/{n_steps}  t={self.t:.3e} s", flush=True)
