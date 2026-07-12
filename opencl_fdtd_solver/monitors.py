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
    Near-to-Far-Field monitor running 100% on the GPU.
    Allocates and accumulates DFT buffers in GPU memory, avoiding host transfers.
    """
    
    def __init__(self, fdtd, center, size, freq):
        super().__init__(fdtd, center, size, freq)
        
        # Allocate GPU buffers for float2 complex arrays (8 bytes per complex element)
        size_bytes = fdtd.size * 8
        mf = cl.mem_flags
        
        self.Ex_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, size_bytes)
        self.Ey_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, size_bytes)
        self.Ez_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, size_bytes)
        self.Hx_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, size_bytes)
        self.Hy_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, size_bytes)
        self.Hz_dft_buf = cl.Buffer(fdtd.ctx, mf.READ_WRITE, size_bytes)

        # Zero out the DFT buffers on GPU
        zeros = np.zeros(fdtd.size * 2, dtype=np.float32)
        cl.enqueue_copy(fdtd.queue, self.Ex_dft_buf, zeros)
        cl.enqueue_copy(fdtd.queue, self.Ey_dft_buf, zeros)
        cl.enqueue_copy(fdtd.queue, self.Ez_dft_buf, zeros)
        cl.enqueue_copy(fdtd.queue, self.Hx_dft_buf, zeros)
        cl.enqueue_copy(fdtd.queue, self.Hy_dft_buf, zeros)
        cl.enqueue_copy(fdtd.queue, self.Hz_dft_buf, zeros)

        fdtd._monitors.append(self)

    def __call__(self, fdtd):
        phase = np.exp(1j * self.omega * fdtd.t) * fdtd.dt
        phase_real = np.float32(phase.real)
        phase_imag = np.float32(phase.imag)

        # Work size for bounding box
        x_dim = self.ix1 - self.ix0 + 1
        y_dim = self.iy1 - self.iy0 + 1
        z_dim = self.iz1 - self.iz0 + 1

        def _accumulate(field_buf, dft_buf):
            fdtd.kern_accumulate_dft(
                fdtd.queue,
                (x_dim, y_dim, z_dim),
                None,
                np.int32(fdtd.Nx), np.int32(fdtd.Ny), np.int32(fdtd.Nz),
                np.int32(self.ix0), np.int32(self.ix1),
                np.int32(self.iy0), np.int32(self.iy1),
                np.int32(self.iz0), np.int32(self.iz1),
                phase_real, phase_imag,
                field_buf,
                dft_buf
            )

        _accumulate(fdtd.Ex_buf, self.Ex_dft_buf)
        _accumulate(fdtd.Ey_buf, self.Ey_dft_buf)
        _accumulate(fdtd.Ez_buf, self.Ez_dft_buf)
        _accumulate(fdtd.Hx_buf, self.Hx_dft_buf)
        _accumulate(fdtd.Hy_buf, self.Hy_dft_buf)
        _accumulate(fdtd.Hz_buf, self.Hz_dft_buf)

    def fetch_dft_fields(self):
        """Fetch DFT fields from GPU to host memory to prepare for farfield calculation."""
        size = self.fdtd.Nx * self.fdtd.Ny * self.fdtd.Nz

        def _fetch(buf):
            host_flat = np.empty(size * 2, dtype=np.float32)
            cl.enqueue_copy(self.fdtd.queue, host_flat, buf)
            # Reconstruct complex numbers from interleaved real/imag
            return (host_flat[0::2] + 1j * host_flat[1::2]).reshape((self.fdtd.Nx, self.fdtd.Ny, self.fdtd.Nz))

        self.Ex_dft = _fetch(self.Ex_dft_buf)
        self.Ey_dft = _fetch(self.Ey_dft_buf)
        self.Ez_dft = _fetch(self.Ez_dft_buf)
        self.Hx_dft = _fetch(self.Hx_dft_buf)
        self.Hy_dft = _fetch(self.Hy_dft_buf)
        self.Hz_dft = _fetch(self.Hz_dft_buf)
