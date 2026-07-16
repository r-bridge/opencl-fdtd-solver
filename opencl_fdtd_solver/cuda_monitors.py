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

"""CUDA near-to-far monitor mirroring :class:`OpenCLNear2FarMonitor`."""

import cupy as cp
import numpy as np

from .constants import C0, ETA0
from .monitors import Near2FarBase, _poynting_db


class CUDANear2FarMonitor(Near2FarBase):
    """
    Near-to-far monitor with face-packed DFT on the GPU (CUDA).

    - Per-step accumulation writes only the 6 Huygens faces (not the full volume).
    - ``get_farfield`` / ``get_farfields`` run the surface integral on CUDA.
    - ``fetch_dft_fields`` downloads face packs only and scatters into sparse
      host volumes for debugging / NumPy-path comparison.

    Precision follows the parent :class:`CUDAFDTD` dtype (float32 or float64).
    """

    def __init__(self, fdtd, center, size, freq):
        super().__init__(fdtd, center, size, freq)

        self.real = fdtd.real
        self._cdtype = fdtd.complex_dtype

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

        # real2 face DFT accumulators, stored as flat (n*2,) real arrays.
        def _dft_buf():
            return cp.zeros(self.n_face_samples * 2, dtype=fdtd.dtype)

        self.Ex_dft_buf = _dft_buf()
        self.Ey_dft_buf = _dft_buf()
        self.Ez_dft_buf = _dft_buf()
        self.Hx_dft_buf = _dft_buf()
        self.Hy_dft_buf = _dft_buf()
        self.Hz_dft_buf = _dft_buf()

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
        self._dft_snap = tuple(
            cp.zeros(self.n_face_samples * 2, dtype=self.fdtd.dtype) for _ in range(6)
        )

    def snapshot_dft(self) -> None:
        """Copy current face DFT buffers → device snapshot (no host round-trip)."""
        self._ensure_dft_snapshot()
        for cur, prev in zip(self._dft_bufs(), self._dft_snap):
            cp.copyto(prev, cur)

    def dft_relative_change(self) -> float:
        """
        CUDA L2 relative change of face DFT vs last ``snapshot_dft``.

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

        if self._dft_rel_n_groups < n_groups:
            self._dft_rel_n_groups = n_groups
            self._dft_rel_partial_num = cp.empty(n_groups, dtype=fdtd.dtype)
            self._dft_rel_partial_den = cp.empty(n_groups, dtype=fdtd.dtype)

        cur = self._dft_bufs()
        prev = self._dft_snap
        item = fdtd.dtype.itemsize
        fdtd.kern_dft_rel_change_partial(
            (n_groups, 1, 1),
            (lsize, 1, 1),
            (np.int32(n), *cur, *prev, self._dft_rel_partial_num, self._dft_rel_partial_den),
            shared_mem=2 * lsize * item,
        )
        num_h = cp.asnumpy(self._dft_rel_partial_num[:n_groups])
        den_h = cp.asnumpy(self._dft_rel_partial_den[:n_groups])
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
        pr_e = self.real(self._phase.real)
        pi_e = self.real(self._phase.imag)
        pr_h = self.real(phase_h.real)
        pi_h = self.real(phase_h.imag)
        o0, o1, o2, o3, o4, o5 = self._offs_i32
        ix0, ix1, iy0, iy1, iz0, iz1 = self._box_i32

        lsize = 128
        grid = ((self.n_face_samples + lsize - 1) // lsize, 1, 1)
        fdtd.kern_accumulate_dft_faces_fused(
            grid,
            (lsize, 1, 1),
            (
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
            ),
        )

    def _ensure_obs_bufs(self, n_obs: int) -> None:
        fdtd = self.fdtd
        if self._obs_buf is None or n_obs > self._obs_cap:
            self._obs_cap = max(n_obs, 64)
            self._obs_buf = cp.empty(self._obs_cap * 3, dtype=fdtd.dtype)
        if self._eh_buf is None or n_obs > self._eh_cap:
            self._eh_cap = max(n_obs, 64)
            # 6 real2 per observation (E,H)
            self._eh_buf = cp.empty(self._eh_cap * 6 * 2, dtype=fdtd.dtype)
            # Integrated N,L (also 6 real2 per obs)
            self._nl_buf = cp.empty(self._eh_cap * 6 * 2, dtype=fdtd.dtype)

    def get_farfields(self, obs_points) -> np.ndarray:
        """
        GPU far-field at many observation points.

        Face samples are reduced in parallel (block reduction + atomics);
        observation angles run as the second launch dimension.

        Returns shape ``(n_obs, 6)`` complex (Ex,Ey,Ez,Hx,Hy,Hz).
        """
        pts = np.asarray(obs_points, dtype=self.fdtd.dtype)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        if pts.shape[1] != 3:
            raise ValueError("obs_points must be (n,3) or (3,)")
        n_obs = int(pts.shape[0])
        n_face = int(self.n_face_samples)
        self._ensure_obs_bufs(n_obs)
        fdtd = self.fdtd
        self._obs_buf[: n_obs * 3].set(np.ascontiguousarray(pts.reshape(-1)))

        # Zero N,L accumulators
        self._nl_buf[: n_obs * 6 * 2].fill(0)

        k_wave = self.real(self.omega / C0)
        dl = self.real(self.dl)
        eta0 = self.real(ETA0)
        offs = [np.int32(o) for o in self._face_offsets]

        # Prefer 256 threads along the face axis (power-of-two local reduce).
        lsize0 = 256
        while lsize0 > 1 and lsize0 > n_face:
            lsize0 //= 2
        lsize0 = max(1, lsize0)
        n_groups0 = (n_face + lsize0 - 1) // lsize0
        shared_bytes = 6 * lsize0 * 2 * self.fdtd.dtype.itemsize  # real2 scratch

        fdtd.kern_farfield_accumulate_nl(
            (n_groups0, n_obs, 1),
            (lsize0, 1, 1),
            (
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
            ),
            shared_mem=shared_bytes,
        )

        fdtd.kern_farfield_nl_to_eh(
            ((n_obs + 63) // 64, 1, 1),
            (64, 1, 1),
            (
                np.int32(n_obs),
                self._obs_buf,
                k_wave,
                eta0,
                self._nl_buf,
                self._eh_buf,
            ),
        )
        host = cp.asnumpy(self._eh_buf[: n_obs * 6 * 2])
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
        pts = np.column_stack([R * np.sin(rad), np.zeros(n), R * np.cos(rad)])
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
            host = cp.asnumpy(buf)
            return host[0::2] + 1j * host[1::2]

        faces = {
            "Ex": _fetch_faces(self.Ex_dft_buf),
            "Ey": _fetch_faces(self.Ey_dft_buf),
            "Ez": _fetch_faces(self.Ez_dft_buf),
            "Hx": _fetch_faces(self.Hx_dft_buf),
            "Hy": _fetch_faces(self.Hy_dft_buf),
            "Hz": _fetch_faces(self.Hz_dft_buf),
        }

        def _scatter(face_arr):
            vol = np.zeros(shape, dtype=self._cdtype)
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
