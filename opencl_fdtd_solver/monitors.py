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
    """Base class for near-to-far monitors: Huygens box layout and host integration.

    Tangential DFT samples are stored face-packed, one complex array per field
    component, in the face order x0, x1, y0, y1, z0, z1. Within a face the
    samples sit at the cell centres of that face, ordered ``(j, k)`` on x
    faces, ``(i, k)`` on y faces and ``(i, j)`` on z faces (last index fastest).
    """

    def __init__(self, fdtd, center, size, freq):
        self.fdtd = fdtd
        freq = float(freq)
        if not np.isfinite(freq) or freq <= 0.0:
            raise ValueError(f"Monitor frequency must be finite and positive; got {freq!r}")
        self.freq = freq
        self.omega = 2.0 * np.pi * freq
        self.dl = fdtd.dl

        cx, cy, cz = (float(v) for v in center)
        sx, sy, sz = (float(v) for v in size)
        if not all(np.isfinite(v) for v in (cx, cy, cz, sx, sy, sz)):
            raise ValueError("Monitor center and size must contain finite numbers")
        if sx <= 0.0 or sy <= 0.0 or sz <= 0.0:
            raise ValueError("Monitor size must be positive along all three axes")

        def _idx(phys, dl):
            # Nearest grid node, with a half-way position going to the higher
            # node at both ends of the box: a box that spans a whole number of
            # cells keeps that extent wherever its centre falls. (Python's
            # round() sends halves to the even node, which made the extent
            # depend on the parity of the centre index.) The 1e-9 absorbs
            # floating-point error in phys / dl at a half-way position.
            return int(np.floor(phys / dl + 0.5 + 1e-9))

        self.ix0 = max(0, _idx(cx - sx / 2, fdtd.dl))
        self.ix1 = min(fdtd.Nx - 1, _idx(cx + sx / 2, fdtd.dl))
        self.iy0 = max(0, _idx(cy - sy / 2, fdtd.dl))
        self.iy1 = min(fdtd.Ny - 1, _idx(cy + sy / 2, fdtd.dl))
        self.iz0 = max(0, _idx(cz - sz / 2, fdtd.dl))
        self.iz1 = min(fdtd.Nz - 1, _idx(cz + sz / 2, fdtd.dl))

        if self.ix1 <= self.ix0 or self.iy1 <= self.iy0 or self.iz1 <= self.iz0:
            raise ValueError(
                f"Near2Far box is degenerate after clamping to grid "
                f"({fdtd.Nx}, {fdtd.Ny}, {fdtd.Nz}): "
                f"x=[{self.ix0}, {self.ix1}], y=[{self.iy0}, {self.iy1}], z=[{self.iz0}, {self.iz1}]"
            )

        self._faces = [
            ("x", self.ix0, -1),
            ("x", self.ix1, +1),
            ("y", self.iy0, -1),
            ("y", self.iy1, +1),
            ("z", self.iz0, -1),
            ("z", self.iz1, +1),
        ]
        self._box = (self.ix0, self.ix1, self.iy0, self.iy1, self.iz0, self.iz1)

        # Packed face layout: x0,x1,y0,y1,z0,z1
        self.nxf = self.ix1 - self.ix0
        self.nyf = self.iy1 - self.iy0
        self.nzf = self.iz1 - self.iz0
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

        # Host face-packed DFT, one array of n_face_samples per component.
        # None until a subclass allocates or downloads them.
        self.Ex_dft_f = None
        self.Ey_dft_f = None
        self.Ez_dft_f = None
        self.Hx_dft_f = None
        self.Hy_dft_f = None
        self.Hz_dft_f = None

        # Sparse host volumes for inspection; only
        # OpenCLNear2FarMonitor.fetch_dft_fields fills them.
        self.Ex_dft = None
        self.Ey_dft = None
        self.Ez_dft = None
        self.Hx_dft = None
        self.Hy_dft = None
        self.Hz_dft = None

    def get_farfield(self, obs_point):
        """
        Far field (Ex, Ey, Ez, Hx, Hy, Hz) at ``obs_point`` (metres) on the host.

        Integrates the equivalent surface currents of the host face-packed DFT
        (``Ex_dft_f`` … ``Hz_dft_f``) with closed-box midpoint quadrature.
        """
        if self.Ex_dft_f is None:
            raise RuntimeError(
                "DFT fields not initialized. Run the simulation and fetch results first."
            )

        dl = self.dl
        k = self.omega / C0
        ox, oy, oz = (float(v) for v in obs_point)
        r = np.sqrt(ox**2 + oy**2 + oz**2)
        rhat = np.array([ox / r, oy / r, oz / r])
        ix0, ix1, iy0, iy1, iz0, iz1 = self._box

        # exp(-j k rhat.r') factorises per axis: cell centres along the two
        # in-face axes, the face plane along the normal.
        centre_phase = (
            np.exp(-1j * k * rhat[0] * (np.arange(ix0, ix1) + 0.5) * dl),
            np.exp(-1j * k * rhat[1] * (np.arange(iy0, iy1) + 0.5) * dl),
            np.exp(-1j * k * rhat[2] * (np.arange(iz0, iz1) + 0.5) * dl),
        )
        fields = {
            "Ex": self.Ex_dft_f,
            "Ey": self.Ey_dft_f,
            "Ez": self.Ez_dft_f,
            "Hx": self.Hx_dft_f,
            "Hy": self.Hy_dft_f,
            "Hz": self.Hz_dft_f,
        }
        # N integrates H x n and L integrates n x E (PHYSICS.md §8). Per face
        # axis, for the outward normal +axis: (vector component, H field, sign)
        # of H x n. n x E = -(E x n), so each L term is the same entry with
        # H replaced by E and the sign reversed.
        h_cross_n = {
            "x": ((1, "Hz", +1), (2, "Hy", -1)),
            "y": ((0, "Hz", -1), (2, "Hx", +1)),
            "z": ((0, "Hy", +1), (1, "Hx", -1)),
        }

        N = np.zeros(3, dtype=np.complex128)
        L = np.zeros(3, dtype=np.complex128)
        for face_id, (face_axis, face_idx, normal_sign) in enumerate(self._faces):
            axis = "xyz".index(face_axis)
            u, v = (a for a in range(3) if a != axis)
            plane_phase = np.exp(-1j * k * rhat[axis] * face_idx * dl)
            phase = (plane_phase * np.outer(centre_phase[u], centre_phase[v])).ravel()
            start = self._face_offsets[face_id]
            sl = slice(start, start + self._face_counts[face_id])
            for comp, h_name, sign in h_cross_n[face_axis]:
                e_name = "E" + h_name[1]
                N[comp] += normal_sign * sign * np.sum(fields[h_name][sl] * phase)
                L[comp] -= normal_sign * sign * np.sum(fields[e_name][sl] * phase)
        N *= dl * dl
        L *= dl * dl

        prefactor = -1j * k / (4.0 * np.pi * r) * np.exp(1j * k * r)
        rxL = np.cross(rhat, L)
        N_t = N - np.dot(rhat, N) * rhat

        E_far = prefactor * (ETA0 * N_t - rxL)
        H_far = np.cross(rhat, E_far) / ETA0

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
    phase, and closed-box midpoint quadrature in :meth:`get_farfield`
    (inherited from :class:`Near2FarBase`).
    """

    def __init__(self, fdtd, center, size, freq):
        super().__init__(fdtd, center, size, freq)
        complex_dtype = getattr(
            fdtd, "accreal_complex_dtype", np.result_type(fdtd.dtype, np.complex64)
        )
        z = np.zeros(self.n_face_samples, dtype=complex_dtype)
        self.Ex_dft_f = z.copy()
        self.Ey_dft_f = z.copy()
        self.Ez_dft_f = z.copy()
        self.Hx_dft_f = z.copy()
        self.Hy_dft_f = z.copy()
        self.Hz_dft_f = z.copy()
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
                    * (Hx[i, j, kk] + Hx[i, j, k_lo] + Hx[i1, j, kk] + Hx[i1, j, k_lo])
                )
                self.Hy_dft_f[fi] += (
                    phase_h
                    * 0.25
                    * (Hy[i, j, kk] + Hy[i, j, k_lo] + Hy[i, j1, kk] + Hy[i, j1, k_lo])
                )


class OpenCLNear2FarMonitor(Near2FarBase):
    """
    Near-to-far monitor with face-packed DFT on the GPU.

    - Per-step accumulation writes only the 6 Huygens faces (not the full volume).
    - ``get_farfield`` / ``get_farfields`` run the surface integral on OpenCL.
    - ``fetch_dft_fields`` downloads the face packs for the host integral
      (``Near2FarBase.get_farfield``) and scatters them into sparse host
      volumes for inspection.
    """

    def __init__(self, fdtd, center, size, freq):
        super().__init__(fdtd, center, size, freq)

        self._real = fdtd.real
        self._dtype = fdtd.dtype
        self._complex_dtype = fdtd.complex_dtype
        self._itemsize = int(fdtd.dtype.itemsize)
        # Accumulator precision (DFT face buffers + N/L reduction): decoupled
        # from the volumetric field dtype above — see EngineFDTD(accum_fp64=).
        self._acc = fdtd.accreal
        self._acc_dtype = np.dtype(fdtd.accreal)
        self._acc_itemsize = int(self._acc_dtype.itemsize)
        self._acc_complex_dtype = fdtd.accreal_complex_dtype

        mf = cl.mem_flags
        nbytes = self.n_face_samples * 2 * self._acc_itemsize  # accreal2
        self.Ex_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, nbytes)
        self.Ey_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, nbytes)
        self.Ez_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, nbytes)
        self.Hx_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, nbytes)
        self.Hy_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, nbytes)
        self.Hz_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, nbytes)

        zeros = np.zeros(self.n_face_samples * 2, dtype=self._acc_dtype)
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

        # H is half a step behind E at monitor time → phase_h = phase_e * exp(-j ω Δt/2).
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
        nbytes = self.n_face_samples * 2 * self._acc_itemsize
        self._dft_snap = tuple(cl.Buffer(fdtd.ctx, mf.READ_WRITE, nbytes) for _ in range(6))
        zeros = np.zeros(self.n_face_samples * 2, dtype=self._acc_dtype)
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
        dev = fdtd.device
        wg_cap = int(getattr(dev, "max_work_group_size", 256) or 256)
        local_mem = int(getattr(dev, "local_mem_size", 32768) or 32768)
        lsize = 256
        while lsize > 1 and (
            lsize > wg_cap or lsize > n or 2 * lsize * self._acc_itemsize > local_mem
        ):
            lsize //= 2
        lsize = max(1, lsize)
        n_groups = (n + lsize - 1) // lsize
        gsize = n_groups * lsize

        mf = cl.mem_flags
        if self._dft_rel_n_groups < n_groups:
            self._dft_rel_n_groups = n_groups
            self._dft_rel_partial_num = cl.Buffer(
                fdtd.ctx, mf.WRITE_ONLY, n_groups * self._acc_itemsize
            )
            self._dft_rel_partial_den = cl.Buffer(
                fdtd.ctx, mf.WRITE_ONLY, n_groups * self._acc_itemsize
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
            cl.LocalMemory(lsize * self._acc_itemsize),
            cl.LocalMemory(lsize * self._acc_itemsize),
        )
        cl.enqueue_nd_range_kernel(
            fdtd.queue,
            fdtd.kern_dft_rel_change_partial,
            (gsize,),
            (lsize,),
        )
        num_h = np.empty(n_groups, dtype=self._acc_dtype)
        den_h = np.empty(n_groups, dtype=self._acc_dtype)
        cl.enqueue_copy(fdtd.queue, num_h, self._dft_rel_partial_num)
        cl.enqueue_copy(fdtd.queue, den_h, self._dft_rel_partial_den)
        fdtd.queue.finish()
        num = float(np.sum(num_h, dtype=np.float64))
        den = float(np.sum(den_h, dtype=np.float64))
        if den <= 0.0 or not np.isfinite(den) or not np.isfinite(num):
            return 1.0
        return float(np.sqrt(max(num, 0.0) / den))

    def __call__(self, fdtd):
        # Evaluate the phase from fdtd.t at every step: a running product
        # phase *= exp(j ω Δt) would accumulate rounding over long runs.
        self._phase = np.exp(1j * self.omega * float(fdtd.t)) * float(fdtd.dt)
        phase_h = self._phase * self._h_half_step
        # The phase is complex128 on the host; cast it to the accumulator's
        # precision, not the volumetric field's, so no fp32 rounding enters here.
        pr_e = self._acc(self._phase.real)
        pi_e = self._acc(self._phase.imag)
        pr_h = self._acc(phase_h.real)
        pi_h = self._acc(phase_h.imag)
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
            self._obs_buf = cl.Buffer(
                fdtd.ctx, mf.READ_ONLY, self._obs_cap * 3 * self._acc_itemsize
            )
        if self._eh_buf is None or n_obs > self._eh_cap:
            self._eh_cap = max(n_obs, 64)
            # 6 accreal2 per observation (E,H) — N/L reduction + final EH
            # both accumulate/carry accreal precision now (see accum_fp64).
            self._eh_buf = cl.Buffer(
                fdtd.ctx, mf.WRITE_ONLY, self._eh_cap * 6 * 2 * self._acc_itemsize
            )
            # Integrated N,L (also 6 accreal2 per obs)
            self._nl_buf = cl.Buffer(
                fdtd.ctx, mf.READ_WRITE, self._eh_cap * 6 * 2 * self._acc_itemsize
            )

    def get_farfields(self, obs_points) -> np.ndarray:
        """
        GPU far-field at many observation points.

        Face samples are reduced in parallel (workgroup reduction + atomics);
        observation angles run as the second launch dimension.

        Returns shape ``(n_obs, 6)`` complex (Ex,Ey,Ez,Hx,Hy,Hz).
        """
        pts = np.asarray(obs_points, dtype=self._acc_dtype)
        if pts.ndim == 1 and pts.size in (0, 3):
            # One point, or an empty list of points.
            pts = pts.reshape(-1, 3)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("obs_points must be (n,3) or (3,)")
        n_obs = int(pts.shape[0])
        if n_obs == 0:
            return np.empty((0, 6), dtype=np.complex128)
        n_face = int(self.n_face_samples)
        self._ensure_obs_bufs(n_obs)
        fdtd = self.fdtd
        flat = np.ascontiguousarray(pts.reshape(-1), dtype=self._acc_dtype)
        cl.enqueue_copy(fdtd.queue, self._obs_buf, flat)

        # Zero N,L accumulators (accreal precision)
        zeros = np.zeros(n_obs * 6 * 2, dtype=self._acc_dtype)
        cl.enqueue_copy(fdtd.queue, self._nl_buf, zeros)

        k_wave = self._acc(self.omega / C0)
        dl = self._acc(self.dl)
        eta0 = self._acc(ETA0)
        offs = [np.int32(o) for o in self._face_offsets]

        # Prefer 256 threads along the face axis, bounded by device work-group
        # and local memory limits (6 * lsize * sizeof(accreal2)).
        dev = fdtd.device
        wg_cap = int(getattr(dev, "max_work_group_size", 256) or 256)
        local_mem = int(getattr(dev, "local_mem_size", 32768) or 32768)
        lsize0 = 256
        while lsize0 > 1 and (
            lsize0 > wg_cap or lsize0 > n_face or 6 * lsize0 * 2 * self._acc_itemsize > local_mem
        ):
            lsize0 //= 2
        lsize0 = max(1, lsize0)
        g0 = ((n_face + lsize0 - 1) // lsize0) * lsize0
        local_bytes = 6 * lsize0 * 2 * self._acc_itemsize  # accreal2 scratch

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
        host = np.empty(n_obs * 6 * 2, dtype=self._acc_dtype)
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
        # The solver derives direction and range from these displacements.
        # Translating the surface coordinates would only add a common phase
        # to E and H, which cancels in the Poynting magnitude.
        displacements = np.column_stack([R * np.sin(rad), np.zeros(n), R * np.cos(rad)])
        eh = self.get_farfields(displacements)
        db = np.empty(n, dtype=np.float64)
        for i in range(n):
            db[i], _ = _poynting_db(eh[i])
        return angles, db

    def fetch_dft_fields(self):
        """
        Download the face-packed DFT (surface samples only) to the host.

        Returns the six face-packed complex arrays keyed by component and
        keeps them as ``Ex_dft_f`` … ``Hz_dft_f``, which
        ``Near2FarBase.get_farfield`` integrates on the host. Prefer
        ``get_farfield`` (GPU) for results.

        For inspection each component is also scattered into a sparse host
        volume ``Ex_dft`` … ``Hz_dft``, indexed by the cell of each face
        sample, from the four faces on which it is tangential. The volumes
        are not a complete copy: along the three box edges through the low
        corner two faces share a cell index, and the volume keeps the sample
        of the face written last (x faces, then y, then z).
        """
        shape = (self.fdtd.Nx, self.fdtd.Ny, self.fdtd.Nz)

        def _fetch_faces(buf):
            host = np.empty(self.n_face_samples * 2, dtype=self._acc_dtype)
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
        self.Ex_dft_f = faces["Ex"]
        self.Ey_dft_f = faces["Ey"]
        self.Ez_dft_f = faces["Ez"]
        self.Hx_dft_f = faces["Hx"]
        self.Hy_dft_f = faces["Hy"]
        self.Hz_dft_f = faces["Hz"]

        def _scatter(face_arr, axes: tuple[str, ...]):
            vol = np.zeros(shape, dtype=self._acc_complex_dtype)
            if "x" in axes:
                for face_id, ii in ((0, self.ix0), (1, self.ix1)):
                    base = self._face_offsets[face_id]
                    chunk = face_arr[base : base + self.nyf * self.nzf].reshape(self.nyf, self.nzf)
                    vol[ii, self.iy0 : self.iy1, self.iz0 : self.iz1] = chunk
            if "y" in axes:
                for face_id, jj in ((2, self.iy0), (3, self.iy1)):
                    base = self._face_offsets[face_id]
                    chunk = face_arr[base : base + self.nxf * self.nzf].reshape(self.nxf, self.nzf)
                    vol[self.ix0 : self.ix1, jj, self.iz0 : self.iz1] = chunk
            if "z" in axes:
                for face_id, kk in ((4, self.iz0), (5, self.iz1)):
                    base = self._face_offsets[face_id]
                    chunk = face_arr[base : base + self.nxf * self.nyf].reshape(self.nxf, self.nyf)
                    vol[self.ix0 : self.ix1, self.iy0 : self.iy1, kk] = chunk
            return vol

        # Only scatter each field component from the 4 faces on which it is
        # tangential, preventing normal-face zeros from clobbering shared edges.
        self.Ex_dft = _scatter(faces["Ex"], ("y", "z"))
        self.Ey_dft = _scatter(faces["Ey"], ("x", "z"))
        self.Ez_dft = _scatter(faces["Ez"], ("x", "y"))
        self.Hx_dft = _scatter(faces["Hx"], ("y", "z"))
        self.Hy_dft = _scatter(faces["Hy"], ("x", "z"))
        self.Hz_dft = _scatter(faces["Hz"], ("x", "y"))
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
    return float(10.0 * np.log10(mag)), mag
