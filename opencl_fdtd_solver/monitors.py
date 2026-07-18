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
import pyopencl as cl

from .constants import C0, ETA0


class Near2FarBase:
    """Base class for Near-to-Far-Field monitors implementing the Huygens surface integration."""

    def __init__(self, fdtd, center, size, freq):
        self.fdtd = fdtd
        self.freq = freq
        self.omega = 2.0 * np.pi * freq
        self.dl = fdtd.dl

        cx, cy, cz = center
        sx, sy, sz = size

        def _idx(phys, dl):
            return int(round(phys / dl))

        self.ix0 = max(0, _idx(cx - sx / 2, fdtd.dl))
        self.ix1 = min(fdtd.Nx - 1, _idx(cx + sx / 2, fdtd.dl))
        self.iy0 = max(0, _idx(cy - sy / 2, fdtd.dl))
        self.iy1 = min(fdtd.Ny - 1, _idx(cy + sy / 2, fdtd.dl))
        self.iz0 = max(0, _idx(cz - sz / 2, fdtd.dl))
        self.iz1 = min(fdtd.Nz - 1, _idx(cz + sz / 2, fdtd.dl))

        self._faces = [
            ("x", self.ix0, -1),
            ("x", self.ix1, +1),
            ("y", self.iy0, -1),
            ("y", self.iy1, +1),
            ("z", self.iz0, -1),
            ("z", self.iz1, +1),
        ]
        self._box = (self.ix0, self.ix1, self.iy0, self.iy1, self.iz0, self.iz1)

        # Placeholders for host DFT fields
        self.Ex_dft = None
        self.Ey_dft = None
        self.Ez_dft = None
        self.Hx_dft = None
        self.Hy_dft = None
        self.Hz_dft = None

    def get_farfield(self, obs_point):
        """
        Compute far-field (Ex, Ey, Ez, Hx, Hy, Hz) at observation point obs_point (metres).
        Uses equivalent surface currents integrated over the Huygens box.
        """
        if self.Ex_dft is None:
            raise RuntimeError(
                "DFT fields not initialized. Run the simulation and fetch results first."
            )

        dl = self.dl
        k = self.omega / C0
        ox, oy, oz = obs_point
        r = np.sqrt(ox**2 + oy**2 + oz**2)
        rx, ry, rz = ox / r, oy / r, oz / r

        Nx_int = Ny_int = Nz_int = 0j
        Lx_int = Ly_int = Lz_int = 0j

        ix0, ix1, iy0, iy1, iz0, iz1 = self._box

        for face_axis, face_idx, normal_sign in self._faces:
            n = normal_sign
            dA = dl * dl

            if face_axis == "x":
                i = face_idx
                js = np.arange(iy0, iy1 + 1)
                ks = np.arange(iz0, iz1 + 1)
                jg, kg = np.meshgrid(js, ks, indexing="ij")
                w = np.ones_like(jg, dtype=np.float64)
                w[(jg == iy0) | (jg == iy1)] *= 0.5
                w[(kg == iz0) | (kg == iz1)] *= 0.5

                xp = i * dl
                yp = (jg + 0.5) * dl
                zp = (kg + 0.5) * dl
                phase_factor = np.exp(1j * k * (rx * xp + ry * yp + rz * zp))

                J_y = n * self.Hz_dft[i, iy0 : iy1 + 1, iz0 : iz1 + 1]
                J_z = -n * self.Hy_dft[i, iy0 : iy1 + 1, iz0 : iz1 + 1]
                M_y = -n * self.Ez_dft[i, iy0 : iy1 + 1, iz0 : iz1 + 1]
                M_z = n * self.Ey_dft[i, iy0 : iy1 + 1, iz0 : iz1 + 1]

                Ny_int += np.sum(J_y * phase_factor * w) * dA
                Nz_int += np.sum(J_z * phase_factor * w) * dA
                Ly_int += np.sum(M_y * phase_factor * w) * dA
                Lz_int += np.sum(M_z * phase_factor * w) * dA

            elif face_axis == "y":
                j = face_idx
                is_ = np.arange(ix0, ix1 + 1)
                ks = np.arange(iz0, iz1 + 1)
                ig, kg = np.meshgrid(is_, ks, indexing="ij")
                w = np.ones_like(ig, dtype=np.float64)
                w[(ig == ix0) | (ig == ix1)] *= 0.5
                w[(kg == iz0) | (kg == iz1)] *= 0.5

                xp = (ig + 0.5) * dl
                yp = j * dl
                zp = (kg + 0.5) * dl
                phase_factor = np.exp(1j * k * (rx * xp + ry * yp + rz * zp))

                J_x = -n * self.Hz_dft[ix0 : ix1 + 1, j, iz0 : iz1 + 1]
                J_z = n * self.Hx_dft[ix0 : ix1 + 1, j, iz0 : iz1 + 1]
                M_x = n * self.Ez_dft[ix0 : ix1 + 1, j, iz0 : iz1 + 1]
                M_z = -n * self.Ex_dft[ix0 : ix1 + 1, j, iz0 : iz1 + 1]

                Nx_int += np.sum(J_x * phase_factor * w) * dA
                Nz_int += np.sum(J_z * phase_factor * w) * dA
                Lx_int += np.sum(M_x * phase_factor * w) * dA
                Lz_int += np.sum(M_z * phase_factor * w) * dA

            else:  # face_axis == 'z'
                kk = face_idx
                is_ = np.arange(ix0, ix1 + 1)
                js = np.arange(iy0, iy1 + 1)
                ig, jg = np.meshgrid(is_, js, indexing="ij")
                w = np.ones_like(ig, dtype=np.float64)
                w[(ig == ix0) | (ig == ix1)] *= 0.5
                w[(jg == iy0) | (jg == iy1)] *= 0.5

                xp = (ig + 0.5) * dl
                yp = (jg + 0.5) * dl
                zp = kk * dl
                phase_factor = np.exp(1j * k * (rx * xp + ry * yp + rz * zp))

                J_x = n * self.Hy_dft[ix0 : ix1 + 1, iy0 : iy1 + 1, kk]
                J_y = -n * self.Hx_dft[ix0 : ix1 + 1, iy0 : iy1 + 1, kk]
                M_x = -n * self.Ey_dft[ix0 : ix1 + 1, iy0 : iy1 + 1, kk]
                M_y = n * self.Ex_dft[ix0 : ix1 + 1, iy0 : iy1 + 1, kk]

                Nx_int += np.sum(J_x * phase_factor * w) * dA
                Ny_int += np.sum(J_y * phase_factor * w) * dA
                Lx_int += np.sum(M_x * phase_factor * w) * dA
                Ly_int += np.sum(M_y * phase_factor * w) * dA

        prefactor = -1j * k / (4.0 * np.pi * r) * np.exp(-1j * k * r)
        N = np.array([Nx_int, Ny_int, Nz_int])
        L = np.array([Lx_int, Ly_int, Lz_int])
        rhat = np.array([rx, ry, rz])

        rxL = np.cross(rhat, L)
        N_t = N - np.dot(rhat, N) * rhat

        # E ∝ η N_⊥ + r̂×L   (Balanis / Taflove far-field equivalence)
        E_far = prefactor * (ETA0 * N_t + rxL)
        H_far = -np.cross(rhat, E_far) / ETA0

        return np.array(
            [
                E_far[0],
                E_far[1],
                E_far[2],
                H_far[0],
                H_far[1],
                H_far[2],
            ]
        )


class NumPyNear2FarMonitor(Near2FarBase):
    """
    Near-to-far monitor with face-packed DFT on the host.

    Matches the OpenCL layout: co-located tangential samples, H half-step
    phase, and trapezoidal edge/corner quadrature in :meth:`get_farfield`.
    """

    def __init__(self, fdtd, center, size, freq):
        super().__init__(fdtd, center, size, freq)
        self.nxf = self.ix1 - self.ix0 + 1
        self.nyf = self.iy1 - self.iy0 + 1
        self.nzf = self.iz1 - self.iz0 + 1
        self._face_counts = (
            self.nyf * self.nzf,
            self.nyf * self.nzf,
            self.nxf * self.nzf,
            self.nxf * self.nzf,
            self.nxf * self.nyf,
            self.nxf * self.nyf,
        )
        offs = [0]
        for c in self._face_counts[:-1]:
            offs.append(offs[-1] + c)
        self._face_offsets = tuple(offs)
        self.n_face_samples = int(sum(self._face_counts))
        z = np.zeros(self.n_face_samples, dtype=np.complex64)
        self.Ex_dft_f = z.copy()
        self.Ey_dft_f = z.copy()
        self.Ez_dft_f = z.copy()
        self.Hx_dft_f = z.copy()
        self.Hy_dft_f = z.copy()
        self.Hz_dft_f = z.copy()
        # Volume placeholders so Near2FarBase.Ex_dft is non-None after a run.
        shape = (fdtd.Nx, fdtd.Ny, fdtd.Nz)
        self.Ex_dft = np.zeros(shape, dtype=np.complex64)
        self.Ey_dft = np.zeros(shape, dtype=np.complex64)
        self.Ez_dft = np.zeros(shape, dtype=np.complex64)
        self.Hx_dft = np.zeros(shape, dtype=np.complex64)
        self.Hy_dft = np.zeros(shape, dtype=np.complex64)
        self.Hz_dft = np.zeros(shape, dtype=np.complex64)
        fdtd.add_monitor(self)

    def __call__(self, fdtd):
        phase_e = np.exp(1j * self.omega * fdtd.t) * fdtd.dt
        phase_h = phase_e * np.exp(-0.5j * self.omega * fdtd.dt)
        ix0, ix1, iy0, iy1, iz0, iz1 = self._box
        Ex, Ey, Ez = np.asarray(fdtd.Ex), np.asarray(fdtd.Ey), np.asarray(fdtd.Ez)
        Hx, Hy, Hz = np.asarray(fdtd.Hx), np.asarray(fdtd.Hy), np.asarray(fdtd.Hz)
        Nx, Ny, Nz = fdtd.Nx, fdtd.Ny, fdtd.Nz
        o0, o1, o2, o3, o4, o5 = self._face_offsets
        nxf, nyf, nzf = self.nxf, self.nyf, self.nzf

        def _avg2(a, b, ok):
            return 0.5 * (a + b) if ok else a

        for i, off in ((ix0, o0), (ix1, o1)):
            for loc in range(nyf * nzf):
                j = iy0 + loc // nzf
                k = iz0 + (loc % nzf)
                k1 = k + 1 if k + 1 < Nz else k
                j1 = j + 1 if j + 1 < Ny else j
                i_lo = i - 1 if i > 0 else i
                fi = off + loc
                self.Ey_dft_f[fi] += phase_e * _avg2(Ey[i, j, k], Ey[i, j, k1], k1 != k)
                self.Ez_dft_f[fi] += phase_e * _avg2(Ez[i, j, k], Ez[i, j1, k], j1 != j)
                self.Hy_dft_f[fi] += (
                    phase_h * 0.25 * (Hy[i, j, k] + Hy[i_lo, j, k] + Hy[i, j1, k] + Hy[i_lo, j1, k])
                )
                self.Hz_dft_f[fi] += (
                    phase_h * 0.25 * (Hz[i, j, k] + Hz[i_lo, j, k] + Hz[i, j, k1] + Hz[i_lo, j, k1])
                )

        for j, off in ((iy0, o2), (iy1, o3)):
            for loc in range(nxf * nzf):
                i = ix0 + loc // nzf
                k = iz0 + (loc % nzf)
                i1 = i + 1 if i + 1 < Nx else i
                k1 = k + 1 if k + 1 < Nz else k
                j_lo = j - 1 if j > 0 else j
                fi = off + loc
                self.Ex_dft_f[fi] += phase_e * _avg2(Ex[i, j, k], Ex[i, j, k1], k1 != k)
                self.Ez_dft_f[fi] += phase_e * _avg2(Ez[i, j, k], Ez[i1, j, k], i1 != i)
                self.Hx_dft_f[fi] += (
                    phase_h * 0.25 * (Hx[i, j, k] + Hx[i, j_lo, k] + Hx[i1, j, k] + Hx[i1, j_lo, k])
                )
                self.Hz_dft_f[fi] += (
                    phase_h * 0.25 * (Hz[i, j, k] + Hz[i, j_lo, k] + Hz[i, j, k1] + Hz[i, j_lo, k1])
                )

        for kk, off in ((iz0, o4), (iz1, o5)):
            for loc in range(nxf * nyf):
                i = ix0 + loc // nyf
                j = iy0 + (loc % nyf)
                i1 = i + 1 if i + 1 < Nx else i
                j1 = j + 1 if j + 1 < Ny else j
                k_lo = kk - 1 if kk > 0 else kk
                fi = off + loc
                self.Ex_dft_f[fi] += phase_e * _avg2(Ex[i, j, kk], Ex[i, j1, kk], j1 != j)
                self.Ey_dft_f[fi] += phase_e * _avg2(Ey[i, j, kk], Ey[i1, j, kk], i1 != i)
                self.Hx_dft_f[fi] += (
                    phase_h
                    * 0.25
                    * (Hx[i, j, kk] + Hx[i, j, k_lo] + Hx[i, j1, kk] + Hx[i, j1, k_lo])
                )
                self.Hy_dft_f[fi] += (
                    phase_h
                    * 0.25
                    * (Hy[i, j, kk] + Hy[i, j, k_lo] + Hy[i1, j, kk] + Hy[i1, j, k_lo])
                )

    def get_farfield(self, obs_point):
        """Far-field from face-packed DFT with trapezoidal face weights."""
        dl = self.dl
        k = self.omega / C0
        ox, oy, oz = obs_point
        r = np.sqrt(ox**2 + oy**2 + oz**2)
        rx, ry, rz = ox / r, oy / r, oz / r
        ix0, ix1, iy0, iy1, iz0, iz1 = self._box
        o0, o1, o2, o3, o4, o5 = self._face_offsets
        nxf, nyf, nzf = self.nxf, self.nyf, self.nzf
        dA = dl * dl
        Nx_int = Ny_int = Nz_int = 0j
        Lx_int = Ly_int = Lz_int = 0j

        def _acc_x(i, off, nf):
            nonlocal Ny_int, Nz_int, Ly_int, Lz_int
            for loc in range(nyf * nzf):
                j = iy0 + loc // nzf
                kk = iz0 + (loc % nzf)
                w = 1.0
                if j == iy0 or j == iy1:
                    w *= 0.5
                if kk == iz0 or kk == iz1:
                    w *= 0.5
                xp, yp, zp = i * dl, (j + 0.5) * dl, (kk + 0.5) * dl
                ph = np.exp(1j * k * (rx * xp + ry * yp + rz * zp))
                fi = off + loc
                Ny_int += nf * self.Hz_dft_f[fi] * ph * dA * w
                Nz_int += -nf * self.Hy_dft_f[fi] * ph * dA * w
                Ly_int += -nf * self.Ez_dft_f[fi] * ph * dA * w
                Lz_int += nf * self.Ey_dft_f[fi] * ph * dA * w

        def _acc_y(j, off, nf):
            nonlocal Nx_int, Nz_int, Lx_int, Lz_int
            for loc in range(nxf * nzf):
                i = ix0 + loc // nzf
                kk = iz0 + (loc % nzf)
                w = 1.0
                if i == ix0 or i == ix1:
                    w *= 0.5
                if kk == iz0 or kk == iz1:
                    w *= 0.5
                xp, yp, zp = (i + 0.5) * dl, j * dl, (kk + 0.5) * dl
                ph = np.exp(1j * k * (rx * xp + ry * yp + rz * zp))
                fi = off + loc
                Nx_int += -nf * self.Hz_dft_f[fi] * ph * dA * w
                Nz_int += nf * self.Hx_dft_f[fi] * ph * dA * w
                Lx_int += nf * self.Ez_dft_f[fi] * ph * dA * w
                Lz_int += -nf * self.Ex_dft_f[fi] * ph * dA * w

        def _acc_z(kk, off, nf):
            nonlocal Nx_int, Ny_int, Lx_int, Ly_int
            for loc in range(nxf * nyf):
                i = ix0 + loc // nyf
                j = iy0 + (loc % nyf)
                w = 1.0
                if i == ix0 or i == ix1:
                    w *= 0.5
                if j == iy0 or j == iy1:
                    w *= 0.5
                xp, yp, zp = (i + 0.5) * dl, (j + 0.5) * dl, kk * dl
                ph = np.exp(1j * k * (rx * xp + ry * yp + rz * zp))
                fi = off + loc
                Nx_int += nf * self.Hy_dft_f[fi] * ph * dA * w
                Ny_int += -nf * self.Hx_dft_f[fi] * ph * dA * w
                Lx_int += -nf * self.Ey_dft_f[fi] * ph * dA * w
                Ly_int += nf * self.Ex_dft_f[fi] * ph * dA * w

        _acc_x(ix0, o0, -1.0)
        _acc_x(ix1, o1, +1.0)
        _acc_y(iy0, o2, -1.0)
        _acc_y(iy1, o3, +1.0)
        _acc_z(iz0, o4, -1.0)
        _acc_z(iz1, o5, +1.0)

        prefactor = -1j * k / (4.0 * np.pi * r) * np.exp(-1j * k * r)
        N = np.array([Nx_int, Ny_int, Nz_int])
        L = np.array([Lx_int, Ly_int, Lz_int])
        rhat = np.array([rx, ry, rz])
        rxL = np.cross(rhat, L)
        N_t = N - np.dot(rhat, N) * rhat
        E_far = prefactor * (ETA0 * N_t + rxL)
        H_far = -np.cross(rhat, E_far) / ETA0
        return np.array([E_far[0], E_far[1], E_far[2], H_far[0], H_far[1], H_far[2]])


class OpenCLNear2FarMonitor(Near2FarBase):
    """
    Near-to-far monitor with face-packed DFT on the GPU.

    - Per-step accumulation writes only the 6 Huygens faces (not the full volume).
    - ``get_farfield`` / ``get_farfields`` run the surface integral on OpenCL.
    - ``fetch_dft_fields`` downloads face packs only and scatters into sparse
      host volumes for debugging / NumPy-path comparison.
    """

    def __init__(self, fdtd, center, size, freq):
        super().__init__(fdtd, center, size, freq)

        self.nxf = self.ix1 - self.ix0 + 1
        self.nyf = self.iy1 - self.iy0 + 1
        self.nzf = self.iz1 - self.iz0 + 1
        # Packed face layout: x0,x1,y0,y1,z0,z1
        self._face_counts = (
            self.nyf * self.nzf,
            self.nyf * self.nzf,
            self.nxf * self.nzf,
            self.nxf * self.nzf,
            self.nxf * self.nyf,
            self.nxf * self.nyf,
        )
        offs = [0]
        for c in self._face_counts[:-1]:
            offs.append(offs[-1] + c)
        self._face_offsets = tuple(offs)
        self.n_face_samples = int(sum(self._face_counts))

        self._real = fdtd.real
        self._dtype = fdtd.dtype
        self._complex_dtype = fdtd.complex_dtype
        self._itemsize = int(fdtd.dtype.itemsize)

        mf = cl.mem_flags
        nbytes = self.n_face_samples * 2 * self._itemsize  # real2
        self.Ex_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, nbytes)
        self.Ey_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, nbytes)
        self.Ez_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, nbytes)
        self.Hx_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, nbytes)
        self.Hy_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, nbytes)
        self.Hz_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, nbytes)

        zeros = np.zeros(self.n_face_samples * 2, dtype=self._dtype)
        for buf in (
            self.Ex_dft_buf,
            self.Ey_dft_buf,
            self.Ez_dft_buf,
            self.Hx_dft_buf,
            self.Hy_dft_buf,
            self.Hz_dft_buf,
        ):
            cl.enqueue_copy(fdtd.queue, buf, zeros)

        self._obs_buf = None
        self._obs_cap = 0
        self._eh_buf = None
        self._nl_buf = None
        self._eh_cap = 0
        # Optional DFT snapshot for relative-change convergence (device-side).
        self._dft_snap = None
        self._dft_rel_partial_num = None
        self._dft_rel_partial_den = None
        self._dft_rel_n_groups = 0

        # Phase recurrence: phase_e_{n+1} = phase_e_n * exp(j ω Δt).
        # H is half a step behind E at monitor time → phase_h = phase_e * exp(-j ω Δt/2).
        self._dphase = np.exp(1j * self.omega * float(fdtd.dt))
        self._h_half_step = np.exp(-0.5j * self.omega * float(fdtd.dt))
        self._phase = None
        self._offs_i32 = tuple(np.int32(o) for o in self._face_offsets)
        self._box_i32 = (
            np.int32(self.ix0),
            np.int32(self.ix1),
            np.int32(self.iy0),
            np.int32(self.iy1),
            np.int32(self.iz0),
            np.int32(self.iz1),
        )
        self._n_face_i32 = np.int32(self.n_face_samples)
        self._nx_i32 = np.int32(fdtd.Nx)
        self._ny_i32 = np.int32(fdtd.Ny)
        self._nz_i32 = np.int32(fdtd.Nz)

        fdtd.add_monitor(self)

    def _dft_bufs(self):
        return (
            self.Ex_dft_buf,
            self.Ey_dft_buf,
            self.Ez_dft_buf,
            self.Hx_dft_buf,
            self.Hy_dft_buf,
            self.Hz_dft_buf,
        )

    def _ensure_dft_snapshot(self) -> None:
        if self._dft_snap is not None:
            return
        mf = cl.mem_flags
        fdtd = self.fdtd
        nbytes = self.n_face_samples * 2 * self._itemsize
        self._dft_snap = tuple(cl.Buffer(fdtd.ctx, mf.READ_WRITE, nbytes) for _ in range(6))
        zeros = np.zeros(self.n_face_samples * 2, dtype=self._dtype)
        for buf in self._dft_snap:
            cl.enqueue_copy(fdtd.queue, buf, zeros)

    def snapshot_dft(self) -> None:
        """Copy current face DFT buffers → device snapshot (no host round-trip)."""
        self._ensure_dft_snapshot()
        q = self.fdtd.queue
        for cur, prev in zip(self._dft_bufs(), self._dft_snap):
            cl.enqueue_copy(q, prev, cur)

    def dft_relative_change(self) -> float:
        """
        OpenCL L2 relative change of face DFT vs last ``snapshot_dft``.

        Returns ``||cur - prev||_2 / ||cur||_2``. If the snapshot is missing or
        ``||cur||≈0``, returns ``1.0`` (treat as not converged).
        """
        if self._dft_snap is None:
            return 1.0
        fdtd = self.fdtd
        n = int(self.n_face_samples)
        lsize = 256
        while lsize > 1 and lsize > n:
            lsize //= 2
        lsize = max(1, lsize)
        n_groups = (n + lsize - 1) // lsize
        gsize = n_groups * lsize

        mf = cl.mem_flags
        if self._dft_rel_n_groups < n_groups:
            self._dft_rel_n_groups = n_groups
            self._dft_rel_partial_num = cl.Buffer(
                fdtd.ctx, mf.WRITE_ONLY, n_groups * self._itemsize
            )
            self._dft_rel_partial_den = cl.Buffer(
                fdtd.ctx, mf.WRITE_ONLY, n_groups * self._itemsize
            )

        cur = self._dft_bufs()
        prev = self._dft_snap
        fdtd.kern_dft_rel_change_partial.set_args(
            np.int32(n),
            cur[0],
            cur[1],
            cur[2],
            cur[3],
            cur[4],
            cur[5],
            prev[0],
            prev[1],
            prev[2],
            prev[3],
            prev[4],
            prev[5],
            self._dft_rel_partial_num,
            self._dft_rel_partial_den,
            cl.LocalMemory(lsize * self._itemsize),
            cl.LocalMemory(lsize * self._itemsize),
        )
        cl.enqueue_nd_range_kernel(
            fdtd.queue,
            fdtd.kern_dft_rel_change_partial,
            (gsize,),
            (lsize,),
        )
        num_h = np.empty(n_groups, dtype=self._dtype)
        den_h = np.empty(n_groups, dtype=self._dtype)
        cl.enqueue_copy(fdtd.queue, num_h, self._dft_rel_partial_num)
        cl.enqueue_copy(fdtd.queue, den_h, self._dft_rel_partial_den)
        fdtd.queue.finish()
        num = float(np.sum(num_h, dtype=np.float64))
        den = float(np.sum(den_h, dtype=np.float64))
        if den <= 0.0 or not np.isfinite(den) or not np.isfinite(num):
            return 1.0
        return float(np.sqrt(max(num, 0.0) / den))

    def __call__(self, fdtd):
        if self._phase is None:
            self._phase = np.exp(1j * self.omega * fdtd.t) * fdtd.dt
        else:
            self._phase *= self._dphase

        phase_h = self._phase * self._h_half_step
        pr_e = self._real(self._phase.real)
        pi_e = self._real(self._phase.imag)
        pr_h = self._real(phase_h.real)
        pi_h = self._real(phase_h.imag)
        o0, o1, o2, o3, o4, o5 = self._offs_i32
        ix0, ix1, iy0, iy1, iz0, iz1 = self._box_i32

        fdtd.kern_accumulate_dft_faces_fused(
            fdtd.queue,
            (self.n_face_samples,),
            None,
            self._nx_i32,
            self._ny_i32,
            self._nz_i32,
            ix0,
            ix1,
            iy0,
            iy1,
            iz0,
            iz1,
            o0,
            o1,
            o2,
            o3,
            o4,
            o5,
            self._n_face_i32,
            pr_e,
            pi_e,
            pr_h,
            pi_h,
            fdtd.Ex_buf,
            fdtd.Ey_buf,
            fdtd.Ez_buf,
            fdtd.Hx_buf,
            fdtd.Hy_buf,
            fdtd.Hz_buf,
            self.Ex_dft_buf,
            self.Ey_dft_buf,
            self.Ez_dft_buf,
            self.Hx_dft_buf,
            self.Hy_dft_buf,
            self.Hz_dft_buf,
        )

    def _ensure_obs_bufs(self, n_obs: int) -> None:
        mf = cl.mem_flags
        fdtd = self.fdtd
        if self._obs_buf is None or n_obs > self._obs_cap:
            self._obs_cap = max(n_obs, 64)
            self._obs_buf = cl.Buffer(fdtd.ctx, mf.READ_ONLY, self._obs_cap * 3 * self._itemsize)
        if self._eh_buf is None or n_obs > self._eh_cap:
            self._eh_cap = max(n_obs, 64)
            # 6 real2 per observation (E,H)
            self._eh_buf = cl.Buffer(fdtd.ctx, mf.WRITE_ONLY, self._eh_cap * 6 * 2 * self._itemsize)
            # Integrated N,L (also 6 real2 per obs)
            self._nl_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, self._eh_cap * 6 * 2 * self._itemsize)

    def get_farfields(self, obs_points) -> np.ndarray:
        """
        GPU far-field at many observation points.

        Face samples are reduced in parallel (workgroup reduction + atomics);
        observation angles run as the second launch dimension.

        Returns shape ``(n_obs, 6)`` complex (Ex,Ey,Ez,Hx,Hy,Hz).
        """
        pts = np.asarray(obs_points, dtype=self._dtype)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        if pts.shape[1] != 3:
            raise ValueError("obs_points must be (n,3) or (3,)")
        n_obs = int(pts.shape[0])
        n_face = int(self.n_face_samples)
        self._ensure_obs_bufs(n_obs)
        fdtd = self.fdtd
        flat = np.ascontiguousarray(pts.reshape(-1), dtype=self._dtype)
        cl.enqueue_copy(fdtd.queue, self._obs_buf, flat)

        # Zero N,L accumulators
        zeros = np.zeros(n_obs * 6 * 2, dtype=self._dtype)
        cl.enqueue_copy(fdtd.queue, self._nl_buf, zeros)

        k_wave = self._real(self.omega / C0)
        dl = self._real(self.dl)
        eta0 = self._real(ETA0)
        offs = [np.int32(o) for o in self._face_offsets]

        # Prefer 256 threads along the face axis (power-of-two local reduce).
        lsize0 = 256
        while lsize0 > 1 and lsize0 > n_face:
            lsize0 //= 2
        lsize0 = max(1, lsize0)
        g0 = ((n_face + lsize0 - 1) // lsize0) * lsize0
        local_bytes = 6 * lsize0 * 2 * self._itemsize  # real2 scratch

        fdtd.kern_farfield_accumulate_nl.set_args(
            np.int32(n_face),
            np.int32(n_obs),
            self._obs_buf,
            k_wave,
            dl,
            np.int32(self.ix0),
            np.int32(self.ix1),
            np.int32(self.iy0),
            np.int32(self.iy1),
            np.int32(self.iz0),
            np.int32(self.iz1),
            offs[0],
            offs[1],
            offs[2],
            offs[3],
            offs[4],
            offs[5],
            self.Ex_dft_buf,
            self.Ey_dft_buf,
            self.Ez_dft_buf,
            self.Hx_dft_buf,
            self.Hy_dft_buf,
            self.Hz_dft_buf,
            self._nl_buf,
            cl.LocalMemory(local_bytes),
        )
        cl.enqueue_nd_range_kernel(
            fdtd.queue,
            fdtd.kern_farfield_accumulate_nl,
            (g0, n_obs),
            (lsize0, 1),
        )

        fdtd.kern_farfield_nl_to_eh(
            fdtd.queue,
            (n_obs,),
            None,
            np.int32(n_obs),
            self._obs_buf,
            k_wave,
            eta0,
            self._nl_buf,
            self._eh_buf,
        )
        host = np.empty(n_obs * 6 * 2, dtype=self._dtype)
        cl.enqueue_copy(fdtd.queue, host, self._eh_buf)
        fdtd.queue.finish()
        c = host[0::2] + 1j * host[1::2]
        return c.reshape(n_obs, 6).astype(np.complex128)

    def get_farfield(self, obs_point):
        """GPU far-field at one observation point (metres)."""
        return self.get_farfields([obs_point])[0]

    def farfield_polar_xz(
        self,
        *,
        distance_m: float = 1000.0,
        n_angles: int = 73,
    ) -> tuple[np.ndarray, np.ndarray]:
        """XZ polar |S| (dB) cut entirely on GPU (download EH only)."""
        R = float(distance_m)
        if R <= 0.0:
            raise ValueError("distance_m must be positive")
        n = max(3, int(n_angles))
        angles = np.linspace(-180.0, 180.0, n, dtype=np.float64)
        rad = np.deg2rad(angles)
        pts = np.column_stack([R * np.sin(rad), np.zeros(n), R * np.cos(rad)]).astype(self._dtype)
        eh = self.get_farfields(pts)
        db = np.empty(n, dtype=np.float64)
        for i in range(n):
            db[i], _ = _poynting_db(eh[i])
        return angles, db

    def fetch_dft_fields(self):
        """
        Download face-packed DFT (~surface only) and scatter into sparse
        host volumes for debugging. Prefer ``get_farfield`` (GPU) instead.
        """
        shape = (self.fdtd.Nx, self.fdtd.Ny, self.fdtd.Nz)

        def _fetch_faces(buf):
            host = np.empty(self.n_face_samples * 2, dtype=self._dtype)
            cl.enqueue_copy(self.fdtd.queue, host, buf)
            return host[0::2] + 1j * host[1::2]

        self.fdtd.queue.finish()
        faces = {
            "Ex": _fetch_faces(self.Ex_dft_buf),
            "Ey": _fetch_faces(self.Ey_dft_buf),
            "Ez": _fetch_faces(self.Ez_dft_buf),
            "Hx": _fetch_faces(self.Hx_dft_buf),
            "Hy": _fetch_faces(self.Hy_dft_buf),
            "Hz": _fetch_faces(self.Hz_dft_buf),
        }

        def _scatter(face_arr):
            vol = np.zeros(shape, dtype=self._complex_dtype)
            # x faces
            for face_id, ii in ((0, self.ix0), (1, self.ix1)):
                base = self._face_offsets[face_id]
                chunk = face_arr[base : base + self.nyf * self.nzf].reshape(self.nyf, self.nzf)
                vol[ii, self.iy0 : self.iy1 + 1, self.iz0 : self.iz1 + 1] = chunk
            # y faces
            for face_id, jj in ((2, self.iy0), (3, self.iy1)):
                base = self._face_offsets[face_id]
                chunk = face_arr[base : base + self.nxf * self.nzf].reshape(self.nxf, self.nzf)
                vol[self.ix0 : self.ix1 + 1, jj, self.iz0 : self.iz1 + 1] = chunk
            # z faces
            for face_id, kk in ((4, self.iz0), (5, self.iz1)):
                base = self._face_offsets[face_id]
                chunk = face_arr[base : base + self.nxf * self.nyf].reshape(self.nxf, self.nyf)
                vol[self.ix0 : self.ix1 + 1, self.iy0 : self.iy1 + 1, kk] = chunk
            return vol

        self.Ex_dft = _scatter(faces["Ex"])
        self.Ey_dft = _scatter(faces["Ey"])
        self.Ez_dft = _scatter(faces["Ez"])
        self.Hx_dft = _scatter(faces["Hx"])
        self.Hy_dft = _scatter(faces["Hy"])
        self.Hz_dft = _scatter(faces["Hz"])
        return faces


def _poynting_mag(ff) -> float:
    E = ff[0:3]
    H = ff[3:6]
    Sx = 0.5 * (E[1] * np.conj(H[2]) - E[2] * np.conj(H[1]))
    Sy = 0.5 * (E[2] * np.conj(H[0]) - E[0] * np.conj(H[2]))
    Sz = 0.5 * (E[0] * np.conj(H[1]) - E[1] * np.conj(H[0]))
    return float(np.sqrt(np.abs(Sx) ** 2 + np.abs(Sy) ** 2 + np.abs(Sz) ** 2))


def _poynting_db(ff) -> tuple[float, float]:
    """
    Return ``(db, |S|)``.

    Absolute |S| from a short DFT is often ≪ 1e-30; do **not** clamp to 1e-30
    before ``log10`` — that flattens every angle to −600 dB and destroys the
    peak-normalized polar pattern. Callers should peak-normalize for display.
    """
    mag = _poynting_mag(ff)
    if not np.isfinite(mag) or mag <= 0.0:
        return float("-inf"), 0.0
    return float(20.0 * np.log10(mag)), mag
