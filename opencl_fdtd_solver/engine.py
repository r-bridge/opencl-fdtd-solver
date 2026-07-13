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


class OpenCLFDTD:
    """
    3D Yee-grid FDTD electromagnetic solver accelerated with OpenCL.
    
    Accepts 3D epsilon array, compiles OpenCL update kernels, and runs the simulation loop.
    Supports pluggable monitors (NumPy and OpenCL models).
    """

    def __init__(self, shape, dl, npml=20, dtype=np.float32, ctx=None, queue=None):
        """
        shape : (Nx, Ny, Nz) Yee cells
        dl    : uniform cell size in metres
        npml  : PML thickness in cells
        dtype : data type for computation (np.float32)
        ctx   : pre-existing OpenCL context (optional)
        queue : pre-existing OpenCL command queue (optional)
        """
        self.Nx, self.Ny, self.Nz = shape
        self.dl = float(dl)
        self.npml = int(npml)
        self.dtype = dtype
        self.t = 0.0
        self.step_num = 0

        # Courant-stable time step
        self.dt = 0.99 * dl / (C0 * np.sqrt(3.0))

        # Setup OpenCL context and queue
        if ctx is None:
            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("No OpenCL platforms found.")
            # Search for a GPU first, then CPU
            devices = []
            for p in platforms:
                devices.extend(p.get_devices(cl.device_type.GPU))
            if not devices:
                for p in platforms:
                    devices.extend(p.get_devices(cl.device_type.CPU))
            if not devices:
                devices = platforms[0].get_devices()
            if not devices:
                raise RuntimeError("No OpenCL devices found.")
            self.device = devices[0]
            self.ctx = cl.Context([self.device])
        else:
            self.ctx = ctx
            self.device = self.ctx.devices[0]

        if queue is None:
            self.queue = cl.CommandQueue(self.ctx)
        else:
            self.queue = queue

        print(f"OpenCL FDTD Solver initialized on device: {self.device.name}")

        # Yee field arrays size
        self.size = self.Nx * self.Ny * self.Nz
        mf = cl.mem_flags

        # Allocate Yee fields on GPU
        self.Ex_buf = cl.Buffer(self.ctx, mf.READ_WRITE, self.size * np.dtype(self.dtype).itemsize)
        self.Ey_buf = cl.Buffer(self.ctx, mf.READ_WRITE, self.size * np.dtype(self.dtype).itemsize)
        self.Ez_buf = cl.Buffer(self.ctx, mf.READ_WRITE, self.size * np.dtype(self.dtype).itemsize)
        self.Hx_buf = cl.Buffer(self.ctx, mf.READ_WRITE, self.size * np.dtype(self.dtype).itemsize)
        self.Hy_buf = cl.Buffer(self.ctx, mf.READ_WRITE, self.size * np.dtype(self.dtype).itemsize)
        self.Hz_buf = cl.Buffer(self.ctx, mf.READ_WRITE, self.size * np.dtype(self.dtype).itemsize)
        self.eps_buf = cl.Buffer(self.ctx, mf.READ_WRITE, self.size * np.dtype(self.dtype).itemsize)

        # Initialize GPU buffers to default values (fields=0, eps_r=1)
        zeros = np.zeros(self.size, dtype=self.dtype)
        ones = np.ones(self.size, dtype=self.dtype)
        cl.enqueue_copy(self.queue, self.Ex_buf, zeros)
        cl.enqueue_copy(self.queue, self.Ey_buf, zeros)
        cl.enqueue_copy(self.queue, self.Ez_buf, zeros)
        cl.enqueue_copy(self.queue, self.Hx_buf, zeros)
        cl.enqueue_copy(self.queue, self.Hy_buf, zeros)
        cl.enqueue_copy(self.queue, self.Hz_buf, zeros)
        cl.enqueue_copy(self.queue, self.eps_buf, ones)

        self._sources = []
        self._monitors = []

        self._build_cpml()
        self._compile_kernels()

    def set_epsilon(self, eps_array):
        """Set the 3D permittivity array on the GPU."""
        assert eps_array.shape == (self.Nx, self.Ny, self.Nz), "Epsilon shape mismatch"
        eps_flat = eps_array.astype(self.dtype).flatten()
        cl.enqueue_copy(self.queue, self.eps_buf, eps_flat)

    def _build_cpml(self):
        """Calculate CPML coefficients and allocate face-local psi buffers on the GPU."""
        dl   = self.dl
        dt   = self.dt
        npml = self.npml
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz

        m          = 3
        sigma_opt  = 0.8 * (m + 1) / (2.0 * ETA0 * dl * npml) if npml > 0 else 0.0
        alpha_max  = 0.05 / ETA0

        def _1d_coeffs(n):
            b = np.ones(n,  dtype=self.dtype)
            c = np.zeros(n, dtype=self.dtype)
            k = np.ones(n,  dtype=self.dtype)
            for i in range(npml):
                for lo, idx in ((True, i), (False, n - npml + i)):
                    xi = (npml - i) / npml if lo else (i + 1) / npml
                    sig   = sigma_opt * xi**m
                    kap   = 1.0
                    alp   = alpha_max * (1.0 - xi)**1
                    decay = (sig / kap + alp) * dt / EPS0
                    b[idx] = np.exp(-decay)
                    denom  = sig + kap * alp
                    c[idx] = 0.0 if denom == 0 else \
                             sig / kap * (b[idx] - 1.0) / denom / dl
                    k[idx] = kap
            return b, c, k

        bx, cx, kx = _1d_coeffs(Nx)
        by, cy, ky = _1d_coeffs(Ny)
        bz, cz, kz = _1d_coeffs(Nz)

        mf = cl.mem_flags
        self.bx_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bx)
        self.cx_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cx)
        self.kx_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kx)

        self.by_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=by)
        self.cy_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cy)
        self.ky_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ky)

        self.bz_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bz)
        self.cz_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cz)
        self.kz_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kz)

        # Face-local psi: only the PML slabs where the corresponding c-coeff is nonzero.
        # x-normal faces: 2*npml * Ny * Nz  (Hy_x, Hz_x, Ey_x, Ez_x)
        # y-normal faces: Nx * 2*npml * Nz  (Hx_y, Hz_y, Ex_y, Ez_y)
        # z-normal faces: Nx * Ny * 2*npml  (Hx_z, Hy_z, Ex_z, Ey_z)
        self.psi_x_size = (2 * npml * Ny * Nz) if npml > 0 else 0
        self.psi_y_size = (Nx * 2 * npml * Nz) if npml > 0 else 0
        self.psi_z_size = (Nx * Ny * 2 * npml) if npml > 0 else 0

        def _psi_buf(n):
            if n == 0:
                # Tiny placeholder so kernel args remain valid if ever referenced.
                return cl.Buffer(self.ctx, mf.READ_WRITE, 4)
            zeros = np.zeros(n, dtype=self.dtype)
            return cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros)

        self.psi_Hy_x_buf = _psi_buf(self.psi_x_size)
        self.psi_Hz_x_buf = _psi_buf(self.psi_x_size)
        self.psi_Ey_x_buf = _psi_buf(self.psi_x_size)
        self.psi_Ez_x_buf = _psi_buf(self.psi_x_size)

        self.psi_Hx_y_buf = _psi_buf(self.psi_y_size)
        self.psi_Hz_y_buf = _psi_buf(self.psi_y_size)
        self.psi_Ex_y_buf = _psi_buf(self.psi_y_size)
        self.psi_Ez_y_buf = _psi_buf(self.psi_y_size)

        self.psi_Hx_z_buf = _psi_buf(self.psi_z_size)
        self.psi_Hy_z_buf = _psi_buf(self.psi_z_size)
        self.psi_Ex_z_buf = _psi_buf(self.psi_z_size)
        self.psi_Ey_z_buf = _psi_buf(self.psi_z_size)

    @staticmethod
    def estimate_device_memory_bytes(shape, npml, dtype=np.float32):
        """Estimated GPU allocation for fields + face-local CPML psi (bytes)."""
        nx, ny, nz = shape
        item = np.dtype(dtype).itemsize
        fields = 7 * nx * ny * nz * item  # Ex..Hz + eps
        if npml <= 0:
            return fields
        psi = (
            4 * (2 * npml * ny * nz)  # x-faces
            + 4 * (nx * 2 * npml * nz)  # y-faces
            + 4 * (nx * ny * 2 * npml)  # z-faces
        ) * item
        # 1D CPML coeff arrays are negligible
        return fields + psi

    def _compile_kernels(self):
        """Compile Yee-grid FDTD update kernels (coalesced NDRange + interior/PML split)."""
        # Work-item mapping: get_global_id(0)=k, (1)=j, (2)=i so adjacent threads
        # touch contiguous addresses along the fastest array axis (k).
        kernel_src = """
        __kernel void update_H_interior(
            int Nx, int Ny, int Nz,
            int npml,
            float dl, float dtm,
            __global const float *Ex,
            __global const float *Ey,
            __global const float *Ez,
            __global float *Hx,
            __global float *Hy,
            __global float *Hz
        ) {
            int k = get_global_id(0) + npml;
            int j = get_global_id(1) + npml;
            int i = get_global_id(2) + npml;

            if (i >= Nx - npml || j >= Ny - npml || k >= Nz - npml) return;

            int idx = i * Ny * Nz + j * Nz + k;
            float inv_dl = 1.0f / dl;

            float dEz_dy = Ez[idx + Nz] - Ez[idx];
            float dEy_dz = Ey[idx + 1] - Ey[idx];
            float dEx_dz = Ex[idx + 1] - Ex[idx];
            float dEz_dx = Ez[idx + Ny * Nz] - Ez[idx];
            float dEy_dx = Ey[idx + Ny * Nz] - Ey[idx];
            float dEx_dy = Ex[idx + Nz] - Ex[idx];

            Hx[idx] -= dtm * (dEz_dy - dEy_dz) * inv_dl;
            Hy[idx] -= dtm * (dEx_dz - dEz_dx) * inv_dl;
            Hz[idx] -= dtm * (dEy_dx - dEx_dy) * inv_dl;
        }

        __kernel void update_H_pml(
            int Nx, int Ny, int Nz,
            int npml,
            float dl, float dtm,
            __global const float *Ex,
            __global const float *Ey,
            __global const float *Ez,
            __global float *Hx,
            __global float *Hy,
            __global float *Hz,
            __global const float *bx, __global const float *cx, __global const float *kx,
            __global const float *by, __global const float *cy, __global const float *ky,
            __global const float *bz, __global const float *cz, __global const float *kz,
            __global float *psi_Hx_y, __global float *psi_Hx_z,
            __global float *psi_Hy_x, __global float *psi_Hy_z,
            __global float *psi_Hz_x, __global float *psi_Hz_y
        ) {
            int k = get_global_id(0);
            int j = get_global_id(1);
            int i = get_global_id(2);

            if (i >= Nx || j >= Ny || k >= Nz) return;

            if (npml > 0 &&
                i >= npml && i < Nx - npml &&
                j >= npml && j < Ny - npml &&
                k >= npml && k < Nz - npml) {
                return;
            }

            int idx = i * Ny * Nz + j * Nz + k;

            float dEz_dy = (j < Ny - 1) ? (Ez[idx + Nz] - Ez[idx]) : 0.0f;
            float dEy_dz = (k < Nz - 1) ? (Ey[idx + 1] - Ey[idx])  : 0.0f;
            float dEx_dz = (k < Nz - 1) ? (Ex[idx + 1] - Ex[idx])  : 0.0f;
            float dEz_dx = (i < Nx - 1) ? (Ez[idx + Ny * Nz] - Ez[idx]) : 0.0f;
            float dEy_dx = (i < Nx - 1) ? (Ey[idx + Ny * Nz] - Ey[idx]) : 0.0f;
            float dEx_dy = (j < Ny - 1) ? (Ex[idx + Nz] - Ex[idx]) : 0.0f;

            int in_x = (i < npml) || (i >= Nx - npml);
            int in_y = (j < npml) || (j >= Ny - npml);
            int in_z = (k < npml) || (k >= Nz - npml);

            float p_Hx_y = 0.0f, p_Hx_z = 0.0f;
            float p_Hy_x = 0.0f, p_Hy_z = 0.0f;
            float p_Hz_x = 0.0f, p_Hz_y = 0.0f;

            if (in_x) {
                int il = (i < npml) ? i : (npml + i - (Nx - npml));
                int xi = il * Ny * Nz + j * Nz + k;
                p_Hy_x = bx[i] * psi_Hy_x[xi] + cx[i] * dEz_dx;
                p_Hz_x = bx[i] * psi_Hz_x[xi] + cx[i] * dEy_dx;
                psi_Hy_x[xi] = p_Hy_x;
                psi_Hz_x[xi] = p_Hz_x;
            }
            if (in_y) {
                int jl = (j < npml) ? j : (npml + j - (Ny - npml));
                int yi = i * (2 * npml) * Nz + jl * Nz + k;
                p_Hx_y = by[j] * psi_Hx_y[yi] + cy[j] * dEz_dy;
                p_Hz_y = by[j] * psi_Hz_y[yi] + cy[j] * dEx_dy;
                psi_Hx_y[yi] = p_Hx_y;
                psi_Hz_y[yi] = p_Hz_y;
            }
            if (in_z) {
                int kl = (k < npml) ? k : (npml + k - (Nz - npml));
                int zi = i * Ny * (2 * npml) + j * (2 * npml) + kl;
                p_Hx_z = bz[k] * psi_Hx_z[zi] + cz[k] * dEy_dz;
                p_Hy_z = bz[k] * psi_Hy_z[zi] + cz[k] * dEx_dz;
                psi_Hx_z[zi] = p_Hx_z;
                psi_Hy_z[zi] = p_Hy_z;
            }

            Hx[idx] -= dtm * (dEz_dy / (ky[j] * dl) + p_Hx_y - dEy_dz / (kz[k] * dl) - p_Hx_z);
            Hy[idx] -= dtm * (dEx_dz / (kz[k] * dl) + p_Hy_z - dEz_dx / (kx[i] * dl) - p_Hy_x);
            Hz[idx] -= dtm * (dEy_dx / (kx[i] * dl) + p_Hz_x - dEx_dy / (ky[j] * dl) - p_Hz_y);
        }

        __kernel void update_E_interior(
            int Nx, int Ny, int Nz,
            int npml,
            float dl, float dt,
            float eps0,
            __global const float *eps_r,
            __global const float *Hx,
            __global const float *Hy,
            __global const float *Hz,
            __global float *Ex,
            __global float *Ey,
            __global float *Ez
        ) {
            int k = get_global_id(0) + npml;
            int j = get_global_id(1) + npml;
            int i = get_global_id(2) + npml;

            if (i >= Nx - npml || j >= Ny - npml || k >= Nz - npml) return;

            int idx = i * Ny * Nz + j * Nz + k;
            float inv_dl = 1.0f / dl;

            float dHz_dy = Hz[idx] - Hz[idx - Nz];
            float dHy_dz = Hy[idx] - Hy[idx - 1];
            float dHx_dz = Hx[idx] - Hx[idx - 1];
            float dHz_dx = Hz[idx] - Hz[idx - Ny * Nz];
            float dHy_dx = Hy[idx] - Hy[idx - Ny * Nz];
            float dHx_dy = Hx[idx] - Hx[idx - Nz];

            float coeff = dt / (eps0 * eps_r[idx]) * inv_dl;
            Ex[idx] += coeff * (dHz_dy - dHy_dz);
            Ey[idx] += coeff * (dHx_dz - dHz_dx);
            Ez[idx] += coeff * (dHy_dx - dHx_dy);
        }

        __kernel void update_E_pml(
            int Nx, int Ny, int Nz,
            int npml,
            float dl, float dt,
            float eps0,
            __global const float *eps_r,
            __global const float *Hx,
            __global const float *Hy,
            __global const float *Hz,
            __global float *Ex,
            __global float *Ey,
            __global float *Ez,
            __global const float *bx, __global const float *cx, __global const float *kx,
            __global const float *by, __global const float *cy, __global const float *ky,
            __global const float *bz, __global const float *cz, __global const float *kz,
            __global float *psi_Ex_y, __global float *psi_Ex_z,
            __global float *psi_Ey_x, __global float *psi_Ey_z,
            __global float *psi_Ez_x, __global float *psi_Ez_y
        ) {
            int k = get_global_id(0);
            int j = get_global_id(1);
            int i = get_global_id(2);

            if (i >= Nx || j >= Ny || k >= Nz) return;

            if (npml > 0 &&
                i >= npml && i < Nx - npml &&
                j >= npml && j < Ny - npml &&
                k >= npml && k < Nz - npml) {
                return;
            }

            int idx = i * Ny * Nz + j * Nz + k;

            float dHz_dy = (j > 0) ? (Hz[idx] - Hz[idx - Nz]) : 0.0f;
            float dHy_dz = (k > 0) ? (Hy[idx] - Hy[idx - 1])  : 0.0f;
            float dHx_dz = (k > 0) ? (Hx[idx] - Hx[idx - 1])  : 0.0f;
            float dHz_dx = (i > 0) ? (Hz[idx] - Hz[idx - Ny * Nz]) : 0.0f;
            float dHy_dx = (i > 0) ? (Hy[idx] - Hy[idx - Ny * Nz]) : 0.0f;
            float dHx_dy = (j > 0) ? (Hx[idx] - Hx[idx - Nz]) : 0.0f;

            int in_x = (i < npml) || (i >= Nx - npml);
            int in_y = (j < npml) || (j >= Ny - npml);
            int in_z = (k < npml) || (k >= Nz - npml);

            float p_Ex_y = 0.0f, p_Ex_z = 0.0f;
            float p_Ey_x = 0.0f, p_Ey_z = 0.0f;
            float p_Ez_x = 0.0f, p_Ez_y = 0.0f;

            if (in_x) {
                int il = (i < npml) ? i : (npml + i - (Nx - npml));
                int xi = il * Ny * Nz + j * Nz + k;
                p_Ey_x = bx[i] * psi_Ey_x[xi] + cx[i] * dHz_dx;
                p_Ez_x = bx[i] * psi_Ez_x[xi] + cx[i] * dHy_dx;
                psi_Ey_x[xi] = p_Ey_x;
                psi_Ez_x[xi] = p_Ez_x;
            }
            if (in_y) {
                int jl = (j < npml) ? j : (npml + j - (Ny - npml));
                int yi = i * (2 * npml) * Nz + jl * Nz + k;
                p_Ex_y = by[j] * psi_Ex_y[yi] + cy[j] * dHz_dy;
                p_Ez_y = by[j] * psi_Ez_y[yi] + cy[j] * dHx_dy;
                psi_Ex_y[yi] = p_Ex_y;
                psi_Ez_y[yi] = p_Ez_y;
            }
            if (in_z) {
                int kl = (k < npml) ? k : (npml + k - (Nz - npml));
                int zi = i * Ny * (2 * npml) + j * (2 * npml) + kl;
                p_Ex_z = bz[k] * psi_Ex_z[zi] + cz[k] * dHy_dz;
                p_Ey_z = bz[k] * psi_Ey_z[zi] + cz[k] * dHx_dz;
                psi_Ex_z[zi] = p_Ex_z;
                psi_Ey_z[zi] = p_Ey_z;
            }

            float coeff = dt / (eps0 * eps_r[idx]);
            Ex[idx] += coeff * (dHz_dy / (ky[j] * dl) + p_Ex_y - dHy_dz / (kz[k] * dl) - p_Ex_z);
            Ey[idx] += coeff * (dHx_dz / (kz[k] * dl) + p_Ey_z - dHz_dx / (kx[i] * dl) - p_Ey_x);
            Ez[idx] += coeff * (dHy_dx / (kx[i] * dl) + p_Ez_x - dHx_dy / (ky[j] * dl) - p_Ez_y);
        }

        __kernel void add_source_Ex(
            int Nx, int Ny, int Nz,
            int z_src, float amp,
            __global float *Ex
        ) {
            int j = get_global_id(0);
            int i = get_global_id(1);

            if (i >= Nx || j >= Ny) return;

            int idx = i * Ny * Nz + j * Nz + z_src;
            Ex[idx] += amp;
        }

        /* Legacy full-box DFT (volume buffer); prefer accumulate_dft_face. */
        __kernel void accumulate_dft(
            int Nx, int Ny, int Nz,
            int ix0, int ix1,
            int iy0, int iy1,
            int iz0, int iz1,
            float phase_real, float phase_imag,
            __global const float *field,
            __global float2 *field_dft
        ) {
            int k = get_global_id(0);
            int j = get_global_id(1);
            int i = get_global_id(2);

            int x_dim = ix1 - ix0 + 1;
            int y_dim = iy1 - iy0 + 1;
            int z_dim = iz1 - iz0 + 1;

            if (i >= x_dim || j >= y_dim || k >= z_dim) return;

            int abs_i = ix0 + i;
            int abs_j = iy0 + j;
            int abs_k = iz0 + k;

            if (abs_i == ix0 || abs_i == ix1 || abs_j == iy0 || abs_j == iy1 || abs_k == iz0 || abs_k == iz1) {
                int idx = abs_i * Ny * Nz + abs_j * Nz + abs_k;
                float val = field[idx];

                float2 current_dft = field_dft[idx];
                current_dft.x += val * phase_real;
                current_dft.y += val * phase_imag;
                field_dft[idx] = current_dft;
            }
        }

        /*
         * Accumulate DFT onto one Huygens face into a packed float2 buffer.
         * face_id: 0=x0, 1=x1, 2=y0, 3=y1, 4=z0, 5=z1
         * Work size: (u, v) with u along the first face axis, v the second
         *   x-faces: (nzf, nyf) → abs (ix, iy0+v, iz0+u)
         *   y-faces: (nzf, nxf) → abs (ix0+v, iy, iz0+u)
         *   z-faces: (nyf, nxf) → abs (ix0+v, iy0+u, iz)
         */
        __kernel void accumulate_dft_face(
            int Nx, int Ny, int Nz,
            int face_id,
            int ix0, int ix1,
            int iy0, int iy1,
            int iz0, int iz1,
            int face_offset,
            float phase_real, float phase_imag,
            __global const float *field,
            __global float2 *face_dft
        ) {
            int u = get_global_id(0);
            int v = get_global_id(1);
            int abs_i, abs_j, abs_k;
            int nxf = ix1 - ix0 + 1;
            int nyf = iy1 - iy0 + 1;
            int nzf = iz1 - iz0 + 1;
            int face_li;

            if (face_id == 0 || face_id == 1) {
                if (u >= nzf || v >= nyf) return;
                abs_i = (face_id == 0) ? ix0 : ix1;
                abs_j = iy0 + v;
                abs_k = iz0 + u;
                face_li = v * nzf + u;
            } else if (face_id == 2 || face_id == 3) {
                if (u >= nzf || v >= nxf) return;
                abs_i = ix0 + v;
                abs_j = (face_id == 2) ? iy0 : iy1;
                abs_k = iz0 + u;
                face_li = v * nzf + u;
            } else {
                if (u >= nyf || v >= nxf) return;
                abs_i = ix0 + v;
                abs_j = iy0 + u;
                abs_k = (face_id == 4) ? iz0 : iz1;
                face_li = v * nyf + u;
            }

            int idx = abs_i * Ny * Nz + abs_j * Nz + abs_k;
            float val = field[idx];
            int dst = face_offset + face_li;
            float2 cur = face_dft[dst];
            cur.x += val * phase_real;
            cur.y += val * phase_imag;
            face_dft[dst] = cur;
        }

        inline float2 cmul(float2 a, float2 b) {
            return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
        }

        inline void caccum(float2 *acc, float2 val) {
            acc->x += val.x;
            acc->y += val.y;
        }

        /*
         * One work-item per observation point. Loops packed DFT faces and
         * writes Ex,Ey,Ez,Hx,Hy,Hz as float2[6] per point.
         */
        __kernel void farfield_from_faces(
            int n_obs,
            __global const float *obs_xyz,
            float k_wave,
            float dl,
            float eta0,
            int ix0, int ix1,
            int iy0, int iy1,
            int iz0, int iz1,
            int off0, int off1, int off2, int off3, int off4, int off5,
            __global const float2 *Ex_f,
            __global const float2 *Ey_f,
            __global const float2 *Ez_f,
            __global const float2 *Hx_f,
            __global const float2 *Hy_f,
            __global const float2 *Hz_f,
            __global float2 *EH_out
        ) {
            int p = get_global_id(0);
            if (p >= n_obs) return;

            float ox = obs_xyz[3 * p + 0];
            float oy = obs_xyz[3 * p + 1];
            float oz = obs_xyz[3 * p + 2];
            float r = sqrt(ox * ox + oy * oy + oz * oz);
            if (r < 1.0e-30f) r = 1.0e-30f;
            float rx = ox / r, ry = oy / r, rz = oz / r;

            int nxf = ix1 - ix0 + 1;
            int nyf = iy1 - iy0 + 1;
            int nzf = iz1 - iz0 + 1;
            float dA = dl * dl;

            float2 Nx = (float2)(0.0f, 0.0f);
            float2 Ny = (float2)(0.0f, 0.0f);
            float2 Nz = (float2)(0.0f, 0.0f);
            float2 Lx = (float2)(0.0f, 0.0f);
            float2 Ly = (float2)(0.0f, 0.0f);
            float2 Lz = (float2)(0.0f, 0.0f);

            int face_offs[6] = { off0, off1, off2, off3, off4, off5 };
            int face_ns[6] = { -1, 1, -1, 1, -1, 1 };

            for (int face = 0; face < 6; face++) {
                int nsgn = face_ns[face];
                float nf = (float)nsgn;
                int base = face_offs[face];

                if (face == 0 || face == 1) {
                    int abs_i = (face == 0) ? ix0 : ix1;
                    float xp = abs_i * dl;
                    for (int v = 0; v < nyf; v++) {
                        float yp = (iy0 + v) * dl;
                        for (int u = 0; u < nzf; u++) {
                            float zp = (iz0 + u) * dl;
                            int li = base + v * nzf + u;
                            float phase = k_wave * (rx * xp + ry * yp + rz * zp);
                            float2 ph = (float2)(cos(phase), sin(phase));
                            float2 Jy = cmul((float2)( nf * Hz_f[li].x,  nf * Hz_f[li].y), ph);
                            float2 Jz = cmul((float2)(-nf * Hy_f[li].x, -nf * Hy_f[li].y), ph);
                            float2 My = cmul((float2)(-nf * Ez_f[li].x, -nf * Ez_f[li].y), ph);
                            float2 Mz = cmul((float2)( nf * Ey_f[li].x,  nf * Ey_f[li].y), ph);
                            caccum(&Ny, (float2)(Jy.x * dA, Jy.y * dA));
                            caccum(&Nz, (float2)(Jz.x * dA, Jz.y * dA));
                            caccum(&Ly, (float2)(My.x * dA, My.y * dA));
                            caccum(&Lz, (float2)(Mz.x * dA, Mz.y * dA));
                        }
                    }
                } else if (face == 2 || face == 3) {
                    int abs_j = (face == 2) ? iy0 : iy1;
                    float yp = abs_j * dl;
                    for (int v = 0; v < nxf; v++) {
                        float xp = (ix0 + v) * dl;
                        for (int u = 0; u < nzf; u++) {
                            float zp = (iz0 + u) * dl;
                            int li = base + v * nzf + u;
                            float phase = k_wave * (rx * xp + ry * yp + rz * zp);
                            float2 ph = (float2)(cos(phase), sin(phase));
                            float2 Jx = cmul((float2)(-nf * Hz_f[li].x, -nf * Hz_f[li].y), ph);
                            float2 Jz = cmul((float2)( nf * Hx_f[li].x,  nf * Hx_f[li].y), ph);
                            float2 Mx = cmul((float2)( nf * Ez_f[li].x,  nf * Ez_f[li].y), ph);
                            float2 Mz = cmul((float2)(-nf * Ex_f[li].x, -nf * Ex_f[li].y), ph);
                            caccum(&Nx, (float2)(Jx.x * dA, Jx.y * dA));
                            caccum(&Nz, (float2)(Jz.x * dA, Jz.y * dA));
                            caccum(&Lx, (float2)(Mx.x * dA, Mx.y * dA));
                            caccum(&Lz, (float2)(Mz.x * dA, Mz.y * dA));
                        }
                    }
                } else {
                    int abs_k = (face == 4) ? iz0 : iz1;
                    float zp = abs_k * dl;
                    for (int v = 0; v < nxf; v++) {
                        float xp = (ix0 + v) * dl;
                        for (int u = 0; u < nyf; u++) {
                            float yp = (iy0 + u) * dl;
                            int li = base + v * nyf + u;
                            float phase = k_wave * (rx * xp + ry * yp + rz * zp);
                            float2 ph = (float2)(cos(phase), sin(phase));
                            float2 Jx = cmul((float2)( nf * Hy_f[li].x,  nf * Hy_f[li].y), ph);
                            float2 Jy = cmul((float2)(-nf * Hx_f[li].x, -nf * Hx_f[li].y), ph);
                            float2 Mx = cmul((float2)(-nf * Ey_f[li].x, -nf * Ey_f[li].y), ph);
                            float2 My = cmul((float2)( nf * Ex_f[li].x,  nf * Ex_f[li].y), ph);
                            caccum(&Nx, (float2)(Jx.x * dA, Jx.y * dA));
                            caccum(&Ny, (float2)(Jy.x * dA, Jy.y * dA));
                            caccum(&Lx, (float2)(Mx.x * dA, Mx.y * dA));
                            caccum(&Ly, (float2)(My.x * dA, My.y * dA));
                        }
                    }
                }
            }

            /* prefactor = -1j * k / (4 pi r) * exp(1j k r) */
            float ang = k_wave * r;
            float2 eikr = (float2)(cos(ang), sin(ang));
            float scale = k_wave / (4.0f * 3.14159265358979323846f * r);
            float2 pref = cmul((float2)(0.0f, -scale), eikr);

            float2 Nvec[3] = { Nx, Ny, Nz };
            float2 Lvec[3] = { Lx, Ly, Lz };
            float rhat[3] = { rx, ry, rz };

            /* rxN = rhat × N ; rxL = rhat × L */
            float2 rxN[3], rxL[3], Nt[3], Lt[3], E[3], H[3];
            rxN[0] = (float2)(rhat[1] * Nvec[2].x - rhat[2] * Nvec[1].x,
                              rhat[1] * Nvec[2].y - rhat[2] * Nvec[1].y);
            rxN[1] = (float2)(rhat[2] * Nvec[0].x - rhat[0] * Nvec[2].x,
                              rhat[2] * Nvec[0].y - rhat[0] * Nvec[2].y);
            rxN[2] = (float2)(rhat[0] * Nvec[1].x - rhat[1] * Nvec[0].x,
                              rhat[0] * Nvec[1].y - rhat[1] * Nvec[0].y);
            rxL[0] = (float2)(rhat[1] * Lvec[2].x - rhat[2] * Lvec[1].x,
                              rhat[1] * Lvec[2].y - rhat[2] * Lvec[1].y);
            rxL[1] = (float2)(rhat[2] * Lvec[0].x - rhat[0] * Lvec[2].x,
                              rhat[2] * Lvec[0].y - rhat[0] * Lvec[2].y);
            rxL[2] = (float2)(rhat[0] * Lvec[1].x - rhat[1] * Lvec[0].x,
                              rhat[0] * Lvec[1].y - rhat[1] * Lvec[0].y);

            float2 Ndot = (float2)(
                rhat[0] * Nvec[0].x + rhat[1] * Nvec[1].x + rhat[2] * Nvec[2].x,
                rhat[0] * Nvec[0].y + rhat[1] * Nvec[1].y + rhat[2] * Nvec[2].y);
            float2 Ldot = (float2)(
                rhat[0] * Lvec[0].x + rhat[1] * Lvec[1].x + rhat[2] * Lvec[2].x,
                rhat[0] * Lvec[0].y + rhat[1] * Lvec[1].y + rhat[2] * Lvec[2].y);

            for (int c = 0; c < 3; c++) {
                Nt[c] = (float2)(Nvec[c].x - Ndot.x * rhat[c], Nvec[c].y - Ndot.y * rhat[c]);
                Lt[c] = (float2)(Lvec[c].x - Ldot.x * rhat[c], Lvec[c].y - Ldot.y * rhat[c]);
                float2 tE = (float2)(Lt[c].x + eta0 * rxN[c].x, Lt[c].y + eta0 * rxN[c].y);
                float2 tH = (float2)(Nt[c].x - rxL[c].x / eta0, Nt[c].y - rxL[c].y / eta0);
                E[c] = cmul(pref, tE);
                H[c] = cmul(pref, tH);
            }

            int o = 6 * p;
            EH_out[o + 0] = E[0];
            EH_out[o + 1] = E[1];
            EH_out[o + 2] = E[2];
            EH_out[o + 3] = H[0];
            EH_out[o + 4] = H[1];
            EH_out[o + 5] = H[2];
        }
        """
        self.program = cl.Program(self.ctx, kernel_src).build()
        self.kern_update_H_interior = cl.Kernel(self.program, "update_H_interior")
        self.kern_update_H_pml = cl.Kernel(self.program, "update_H_pml")
        self.kern_update_E_interior = cl.Kernel(self.program, "update_E_interior")
        self.kern_update_E_pml = cl.Kernel(self.program, "update_E_pml")
        self.kern_add_source_Ex = cl.Kernel(self.program, "add_source_Ex")
        self.kern_accumulate_dft = cl.Kernel(self.program, "accumulate_dft")
        self.kern_accumulate_dft_face = cl.Kernel(self.program, "accumulate_dft_face")
        self.kern_farfield_from_faces = cl.Kernel(self.program, "farfield_from_faces")

        # Cached launch geometries (coalesced: Nz, Ny, Nx).
        self._gs_full = (self.Nz, self.Ny, self.Nx)
        n = self.npml
        nx_i = self.Nx - 2 * n
        ny_i = self.Ny - 2 * n
        nz_i = self.Nz - 2 * n
        self._gs_interior = (nz_i, ny_i, nx_i) if (nx_i > 0 and ny_i > 0 and nz_i > 0) else None

    def add_source_Ex(self, z_src, amp):
        """Adds a sheet source value directly on the GPU using a kernel."""
        self.kern_add_source_Ex(
            self.queue,
            (self.Ny, self.Nx),
            None,
            np.int32(self.Nx), np.int32(self.Ny), np.int32(self.Nz),
            np.int32(z_src), np.float32(amp),
            self.Ex_buf
        )

    def _update_H(self):
        dtm = self.dt / MU0
        nx, ny, nz = np.int32(self.Nx), np.int32(self.Ny), np.int32(self.Nz)
        npml = np.int32(self.npml)
        dl = np.float32(self.dl)
        dtm_f = np.float32(dtm)

        if self._gs_interior is not None:
            self.kern_update_H_interior(
                self.queue, self._gs_interior, None,
                nx, ny, nz, npml, dl, dtm_f,
                self.Ex_buf, self.Ey_buf, self.Ez_buf,
                self.Hx_buf, self.Hy_buf, self.Hz_buf,
            )

        if self.npml > 0:
            self.kern_update_H_pml(
                self.queue, self._gs_full, None,
                nx, ny, nz, npml, dl, dtm_f,
                self.Ex_buf, self.Ey_buf, self.Ez_buf,
                self.Hx_buf, self.Hy_buf, self.Hz_buf,
                self.bx_buf, self.cx_buf, self.kx_buf,
                self.by_buf, self.cy_buf, self.ky_buf,
                self.bz_buf, self.cz_buf, self.kz_buf,
                self.psi_Hx_y_buf, self.psi_Hx_z_buf,
                self.psi_Hy_x_buf, self.psi_Hy_z_buf,
                self.psi_Hz_x_buf, self.psi_Hz_y_buf,
            )
        elif self._gs_interior is None:
            # npml == 0 and no interior box: update whole domain with PML kernel path disabled.
            # Fall through should not happen for valid grids; treat as full interior.
            self.kern_update_H_interior(
                self.queue, self._gs_full, None,
                nx, ny, nz, np.int32(0), dl, dtm_f,
                self.Ex_buf, self.Ey_buf, self.Ez_buf,
                self.Hx_buf, self.Hy_buf, self.Hz_buf,
            )

    def _update_E(self):
        nx, ny, nz = np.int32(self.Nx), np.int32(self.Ny), np.int32(self.Nz)
        npml = np.int32(self.npml)
        dl = np.float32(self.dl)
        dt = np.float32(self.dt)
        eps0 = np.float32(EPS0)

        if self._gs_interior is not None:
            self.kern_update_E_interior(
                self.queue, self._gs_interior, None,
                nx, ny, nz, npml, dl, dt, eps0,
                self.eps_buf,
                self.Hx_buf, self.Hy_buf, self.Hz_buf,
                self.Ex_buf, self.Ey_buf, self.Ez_buf,
            )

        if self.npml > 0:
            self.kern_update_E_pml(
                self.queue, self._gs_full, None,
                nx, ny, nz, npml, dl, dt, eps0,
                self.eps_buf,
                self.Hx_buf, self.Hy_buf, self.Hz_buf,
                self.Ex_buf, self.Ey_buf, self.Ez_buf,
                self.bx_buf, self.cx_buf, self.kx_buf,
                self.by_buf, self.cy_buf, self.ky_buf,
                self.bz_buf, self.cz_buf, self.kz_buf,
                self.psi_Ex_y_buf, self.psi_Ex_z_buf,
                self.psi_Ey_x_buf, self.psi_Ey_z_buf,
                self.psi_Ez_x_buf, self.psi_Ez_y_buf,
            )
        elif self._gs_interior is None:
            self.kern_update_E_interior(
                self.queue, self._gs_full, None,
                nx, ny, nz, np.int32(0), dl, dt, eps0,
                self.eps_buf,
                self.Hx_buf, self.Hy_buf, self.Hz_buf,
                self.Ex_buf, self.Ey_buf, self.Ez_buf,
            )

    def step(self):
        """Single timestep Yee-update with sources and monitors."""
        self._update_H()
        for src in self._sources:
            src(self)
        self._update_E()
        self.t += self.dt
        self.step_num += 1
        for mon in self._monitors:
            mon(self)

    def run(self, n_steps, progress_every=0):
        """Run simulation for n_steps."""
        for i in range(n_steps):
            self.step()
            if progress_every and i % progress_every == 0:
                print(f"  step {i}/{n_steps}  t={self.t:.3e} s", flush=True)

    # ── CPU Host properties to read Yee fields from GPU (NumPy compatibility) ──
    def _read_field(self, buf):
        host_arr = np.empty((self.Nx, self.Ny, self.Nz), dtype=self.dtype)
        cl.enqueue_copy(self.queue, host_arr, buf)
        return host_arr

    def read_point(self, field_name, i, j, k):
        """Read a single point value from a field buffer on the GPU (efficient 4-byte copy)."""
        buf = getattr(self, f"{field_name}_buf")
        idx = int(i * self.Ny * self.Nz + j * self.Nz + k)
        dest = np.empty(1, dtype=self.dtype)
        cl.enqueue_copy(self.queue, dest, buf, src_offset=idx * 4)
        return float(dest[0])

    @property
    def Ex(self): return self._read_field(self.Ex_buf)
    @property
    def Ey(self): return self._read_field(self.Ey_buf)
    @property
    def Ez(self): return self._read_field(self.Ez_buf)
    @property
    def Hx(self): return self._read_field(self.Hx_buf)
    @property
    def Hy(self): return self._read_field(self.Hy_buf)
    @property
    def Hz(self): return self._read_field(self.Hz_buf)
