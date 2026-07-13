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


C0   = 299_792_458.0
MU0  = 4e-7 * np.pi
EPS0 = 1.0 / (MU0 * C0**2)
ETA0 = np.sqrt(MU0 / EPS0)


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

        self.ix0 = max(0, _idx(cx - sx/2, fdtd.dl))
        self.ix1 = min(fdtd.Nx - 1, _idx(cx + sx/2, fdtd.dl))
        self.iy0 = max(0, _idx(cy - sy/2, fdtd.dl))
        self.iy1 = min(fdtd.Ny - 1, _idx(cy + sy/2, fdtd.dl))
        self.iz0 = max(0, _idx(cz - sz/2, fdtd.dl))
        self.iz1 = min(fdtd.Nz - 1, _idx(cz + sz/2, fdtd.dl))

        self._faces = [
            ('x', self.ix0, -1),
            ('x', self.ix1, +1),
            ('y', self.iy0, -1),
            ('y', self.iy1, +1),
            ('z', self.iz0, -1),
            ('z', self.iz1, +1),
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
            raise RuntimeError("DFT fields not initialized. Run the simulation and fetch results first.")

        dl = self.dl
        k = self.omega / C0
        ox, oy, oz = obs_point
        r = np.sqrt(ox**2 + oy**2 + oz**2)
        rx, ry, rz = ox/r, oy/r, oz/r

        Nx_int = Ny_int = Nz_int = 0j
        Lx_int = Ly_int = Lz_int = 0j

        ix0, ix1, iy0, iy1, iz0, iz1 = self._box

        for face_axis, face_idx, normal_sign in self._faces:
            n = normal_sign
            dA = dl * dl

            if face_axis == 'x':
                i = face_idx
                js = np.arange(iy0, iy1+1)
                ks = np.arange(iz0, iz1+1)
                jg, kg = np.meshgrid(js, ks, indexing='ij')

                xp = i * dl; yp = jg * dl; zp = kg * dl
                phase_factor = np.exp(1j * k * (rx*xp + ry*yp + rz*zp))

                J_y =  n * self.Hz_dft[i, iy0:iy1+1, iz0:iz1+1]
                J_z = -n * self.Hy_dft[i, iy0:iy1+1, iz0:iz1+1]
                M_y = -n * self.Ez_dft[i, iy0:iy1+1, iz0:iz1+1]
                M_z =  n * self.Ey_dft[i, iy0:iy1+1, iz0:iz1+1]

                Ny_int += np.sum(J_y * phase_factor) * dA
                Nz_int += np.sum(J_z * phase_factor) * dA
                Ly_int += np.sum(M_y * phase_factor) * dA
                Lz_int += np.sum(M_z * phase_factor) * dA

            elif face_axis == 'y':
                j = face_idx
                is_ = np.arange(ix0, ix1+1)
                ks  = np.arange(iz0, iz1+1)
                ig, kg = np.meshgrid(is_, ks, indexing='ij')

                xp = ig * dl; yp = j * dl; zp = kg * dl
                phase_factor = np.exp(1j * k * (rx*xp + ry*yp + rz*zp))

                J_x = -n * self.Hz_dft[ix0:ix1+1, j, iz0:iz1+1]
                J_z =  n * self.Hx_dft[ix0:ix1+1, j, iz0:iz1+1]
                M_x =  n * self.Ez_dft[ix0:ix1+1, j, iz0:iz1+1]
                M_z = -n * self.Ex_dft[ix0:ix1+1, j, iz0:iz1+1]

                Nx_int += np.sum(J_x * phase_factor) * dA
                Nz_int += np.sum(J_z * phase_factor) * dA
                Lx_int += np.sum(M_x * phase_factor) * dA
                Lz_int += np.sum(M_z * phase_factor) * dA

            else:  # face_axis == 'z'
                kk = face_idx
                is_ = np.arange(ix0, ix1+1)
                js  = np.arange(iy0, iy1+1)
                ig, jg = np.meshgrid(is_, js, indexing='ij')

                xp = ig * dl; yp = jg * dl; zp = kk * dl
                phase_factor = np.exp(1j * k * (rx*xp + ry*yp + rz*zp))

                J_x =  n * self.Hy_dft[ix0:ix1+1, iy0:iy1+1, kk]
                J_y = -n * self.Hx_dft[ix0:ix1+1, iy0:iy1+1, kk]
                M_x = -n * self.Ey_dft[ix0:ix1+1, iy0:iy1+1, kk]
                M_y =  n * self.Ex_dft[ix0:ix1+1, iy0:iy1+1, kk]

                Nx_int += np.sum(J_x * phase_factor) * dA
                Ny_int += np.sum(J_y * phase_factor) * dA
                Lx_int += np.sum(M_x * phase_factor) * dA
                Ly_int += np.sum(M_y * phase_factor) * dA

        prefactor = -1j * k / (4.0 * np.pi * r) * np.exp(1j * k * r)
        N = np.array([Nx_int, Ny_int, Nz_int])
        L = np.array([Lx_int, Ly_int, Lz_int])
        rhat = np.array([rx, ry, rz])

        rxN = np.cross(rhat, N)
        rxL = np.cross(rhat, L)

        N_t = N - np.dot(rhat, N) * rhat
        L_t = L - np.dot(rhat, L) * rhat

        E_far = prefactor * (L_t + ETA0 * rxN)
        H_far = prefactor * (N_t - rxL / ETA0)

        return np.array([
            E_far[0], E_far[1], E_far[2],
            H_far[0], H_far[1], H_far[2],
        ])


class NumPyNear2FarMonitor(Near2FarBase):
    """
    Near-to-Far-Field monitor using NumPy for CPU-based accumulation.
    Fetches the 3D fields from the solver at each step.
    """
    
    def __init__(self, fdtd, center, size, freq):
        super().__init__(fdtd, center, size, freq)
        shape = (fdtd.Nx, fdtd.Ny, fdtd.Nz)
        self.Ex_dft = np.zeros(shape, dtype=np.complex64)
        self.Ey_dft = np.zeros(shape, dtype=np.complex64)
        self.Ez_dft = np.zeros(shape, dtype=np.complex64)
        self.Hx_dft = np.zeros(shape, dtype=np.complex64)
        self.Hy_dft = np.zeros(shape, dtype=np.complex64)
        self.Hz_dft = np.zeros(shape, dtype=np.complex64)
        fdtd._monitors.append(self)

    def __call__(self, fdtd):
        phase = np.exp(1j * self.omega * fdtd.t) * fdtd.dt
        ix0, ix1, iy0, iy1, iz0, iz1 = self._box

        # Only accumulate on box faces
        def _add(dft, field):
            arr = np.asarray(field)
            dft[ix0, iy0:iy1+1, iz0:iz1+1] += phase * arr[ix0, iy0:iy1+1, iz0:iz1+1]
            dft[ix1, iy0:iy1+1, iz0:iz1+1] += phase * arr[ix1, iy0:iy1+1, iz0:iz1+1]
            dft[ix0:ix1+1, iy0, iz0:iz1+1] += phase * arr[ix0:ix1+1, iy0, iz0:iz1+1]
            dft[ix0:ix1+1, iy1, iz0:iz1+1] += phase * arr[ix0:ix1+1, iy1, iz0:iz1+1]
            dft[ix0:ix1+1, iy0:iy1+1, iz0] += phase * arr[ix0:ix1+1, iy0:iy1+1, iz0]
            dft[ix0:ix1+1, iy0:iy1+1, iz1] += phase * arr[ix0:ix1+1, iy0:iy1+1, iz1]

        _add(self.Ex_dft, fdtd.Ex)
        _add(self.Ey_dft, fdtd.Ey)
        _add(self.Ez_dft, fdtd.Ez)
        _add(self.Hx_dft, fdtd.Hx)
        _add(self.Hy_dft, fdtd.Hy)
        _add(self.Hz_dft, fdtd.Hz)


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

        mf = cl.mem_flags
        nbytes = self.n_face_samples * 8  # float2
        self.Ex_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, nbytes)
        self.Ey_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, nbytes)
        self.Ez_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, nbytes)
        self.Hx_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, nbytes)
        self.Hy_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, nbytes)
        self.Hz_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, nbytes)

        zeros = np.zeros(self.n_face_samples * 2, dtype=np.float32)
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
        self._eh_cap = 0

        fdtd._monitors.append(self)

    def __call__(self, fdtd):
        phase = np.exp(1j * self.omega * fdtd.t) * fdtd.dt
        phase_real = np.float32(phase.real)
        phase_imag = np.float32(phase.imag)

        nx = np.int32(fdtd.Nx)
        ny = np.int32(fdtd.Ny)
        nz = np.int32(fdtd.Nz)
        ix0, ix1 = np.int32(self.ix0), np.int32(self.ix1)
        iy0, iy1 = np.int32(self.iy0), np.int32(self.iy1)
        iz0, iz1 = np.int32(self.iz0), np.int32(self.iz1)

        def _acc(field_buf, dft_buf):
            for face_id in range(6):
                if face_id in (0, 1):
                    gsize = (self.nzf, self.nyf)
                elif face_id in (2, 3):
                    gsize = (self.nzf, self.nxf)
                else:
                    gsize = (self.nyf, self.nxf)
                fdtd.kern_accumulate_dft_face(
                    fdtd.queue,
                    gsize,
                    None,
                    nx, ny, nz,
                    np.int32(face_id),
                    ix0, ix1, iy0, iy1, iz0, iz1,
                    np.int32(self._face_offsets[face_id]),
                    phase_real, phase_imag,
                    field_buf,
                    dft_buf,
                )

        _acc(fdtd.Ex_buf, self.Ex_dft_buf)
        _acc(fdtd.Ey_buf, self.Ey_dft_buf)
        _acc(fdtd.Ez_buf, self.Ez_dft_buf)
        _acc(fdtd.Hx_buf, self.Hx_dft_buf)
        _acc(fdtd.Hy_buf, self.Hy_dft_buf)
        _acc(fdtd.Hz_buf, self.Hz_dft_buf)

    def _ensure_obs_bufs(self, n_obs: int) -> None:
        mf = cl.mem_flags
        fdtd = self.fdtd
        if self._obs_buf is None or n_obs > self._obs_cap:
            self._obs_cap = max(n_obs, 64)
            self._obs_buf = cl.Buffer(fdtd.ctx, mf.READ_ONLY, self._obs_cap * 3 * 4)
        if self._eh_buf is None or n_obs > self._eh_cap:
            self._eh_cap = max(n_obs, 64)
            # 6 float2 per observation
            self._eh_buf = cl.Buffer(fdtd.ctx, mf.WRITE_ONLY, self._eh_cap * 6 * 8)

    def get_farfields(self, obs_points) -> np.ndarray:
        """
        GPU far-field at many observation points.

        Returns shape ``(n_obs, 6)`` complex64 (Ex,Ey,Ez,Hx,Hy,Hz).
        """
        pts = np.asarray(obs_points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        if pts.shape[1] != 3:
            raise ValueError("obs_points must be (n,3) or (3,)")
        n_obs = int(pts.shape[0])
        self._ensure_obs_bufs(n_obs)
        fdtd = self.fdtd
        flat = np.ascontiguousarray(pts.reshape(-1), dtype=np.float32)
        cl.enqueue_copy(fdtd.queue, self._obs_buf, flat)

        k_wave = np.float32(self.omega / C0)
        dl = np.float32(self.dl)
        eta0 = np.float32(ETA0)
        offs = [np.int32(o) for o in self._face_offsets]

        fdtd.kern_farfield_from_faces(
            fdtd.queue,
            (n_obs,),
            None,
            np.int32(n_obs),
            self._obs_buf,
            k_wave,
            dl,
            eta0,
            np.int32(self.ix0), np.int32(self.ix1),
            np.int32(self.iy0), np.int32(self.iy1),
            np.int32(self.iz0), np.int32(self.iz1),
            offs[0], offs[1], offs[2], offs[3], offs[4], offs[5],
            self.Ex_dft_buf,
            self.Ey_dft_buf,
            self.Ez_dft_buf,
            self.Hx_dft_buf,
            self.Hy_dft_buf,
            self.Hz_dft_buf,
            self._eh_buf,
        )
        host = np.empty(n_obs * 6 * 2, dtype=np.float32)
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
        pts = np.column_stack(
            [R * np.sin(rad), np.zeros(n), R * np.cos(rad)]
        ).astype(np.float32)
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
            host = np.empty(self.n_face_samples * 2, dtype=np.float32)
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
            vol = np.zeros(shape, dtype=np.complex64)
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


def _poynting_db(ff) -> tuple[float, float]:
    E = ff[0:3]
    H = ff[3:6]
    Sx = 0.5 * (E[1] * np.conj(H[2]) - E[2] * np.conj(H[1]))
    Sy = 0.5 * (E[2] * np.conj(H[0]) - E[0] * np.conj(H[2]))
    Sz = 0.5 * (E[0] * np.conj(H[1]) - E[1] * np.conj(H[0]))
    mag = float(np.sqrt(np.abs(Sx) ** 2 + np.abs(Sy) ** 2 + np.abs(Sz) ** 2))
    db = float(20.0 * np.log10(max(mag, 1e-30)))
    return db, mag
