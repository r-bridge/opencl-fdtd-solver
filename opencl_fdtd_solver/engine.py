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
        """Calculate CPML coefficients and allocate auxiliary buffers on the GPU."""
        dl   = self.dl
        dt   = self.dt
        npml = self.npml
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz

        m          = 3
        sigma_opt  = 0.8 * (m + 1) / (2.0 * ETA0 * dl * npml)
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
        # Copy 1D CPML arrays to GPU
        self.bx_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bx)
        self.cx_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cx)
        self.kx_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kx)

        self.by_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=by)
        self.cy_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cy)
        self.ky_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ky)

        self.bz_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bz)
        self.cz_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cz)
        self.kz_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kz)

        # Allocate CPML 3D correction (psi) arrays
        zeros = np.zeros(self.size, dtype=self.dtype)
        
        self.psi_Hx_y_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros)
        self.psi_Hx_z_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros)
        self.psi_Hy_x_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros)
        self.psi_Hy_z_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros)
        self.psi_Hz_x_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros)
        self.psi_Hz_y_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros)

        self.psi_Ex_y_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros)
        self.psi_Ex_z_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros)
        self.psi_Ey_x_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros)
        self.psi_Ey_z_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros)
        self.psi_Ez_x_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros)
        self.psi_Ez_y_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zeros)

    def _compile_kernels(self):
        """Compile Yee-grid FDTD update kernels."""
        kernel_src = """
        __kernel void update_H(
            int Nx, int Ny, int Nz,
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
            int i = get_global_id(0);
            int j = get_global_id(1);
            int k = get_global_id(2);

            if (i >= Nx || j >= Ny || k >= Nz) return;

            int idx = i * Ny * Nz + j * Nz + k;

            // Forward differences with boundary conditions (matching _fwd axis zero padding)
            float dEz_dy = (j < Ny - 1) ? (Ez[idx + Nz] - Ez[idx]) : 0.0f;
            float dEy_dz = (k < Nz - 1) ? (Ey[idx + 1] - Ey[idx])  : 0.0f;
            float dEx_dz = (k < Nz - 1) ? (Ex[idx + 1] - Ex[idx])  : 0.0f;
            float dEz_dx = (i < Nx - 1) ? (Ez[idx + Ny * Nz] - Ez[idx]) : 0.0f;
            float dEy_dx = (i < Nx - 1) ? (Ey[idx + Ny * Nz] - Ey[idx]) : 0.0f;
            float dEx_dy = (j < Ny - 1) ? (Ex[idx + Nz] - Ex[idx]) : 0.0f;

            // CPML psi updates
            float p_Hx_y = by[j] * psi_Hx_y[idx] + cy[j] * dEz_dy;
            float p_Hx_z = bz[k] * psi_Hx_z[idx] + cz[k] * dEy_dz;
            float p_Hy_x = bx[i] * psi_Hy_x[idx] + cx[i] * dEz_dx;
            float p_Hy_z = bz[k] * psi_Hy_z[idx] + cz[k] * dEx_dz;
            float p_Hz_x = bx[i] * psi_Hz_x[idx] + cx[i] * dEy_dx;
            float p_Hz_y = by[j] * psi_Hz_y[idx] + cy[j] * dEx_dy;

            psi_Hx_y[idx] = p_Hx_y;
            psi_Hx_z[idx] = p_Hx_z;
            psi_Hy_x[idx] = p_Hy_x;
            psi_Hy_z[idx] = p_Hy_z;
            psi_Hz_x[idx] = p_Hz_x;
            psi_Hz_y[idx] = p_Hz_y;

            // Field update
            Hx[idx] -= dtm * (dEz_dy / (ky[j] * dl) + p_Hx_y - dEy_dz / (kz[k] * dl) - p_Hx_z);
            Hy[idx] -= dtm * (dEx_dz / (kz[k] * dl) + p_Hy_z - dEz_dx / (kx[i] * dl) - p_Hy_x);
            Hz[idx] -= dtm * (dEy_dx / (kx[i] * dl) + p_Hz_x - dEx_dy / (ky[j] * dl) - p_Hz_y);
        }

        __kernel void update_E(
            int Nx, int Ny, int Nz,
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
            int i = get_global_id(0);
            int j = get_global_id(1);
            int k = get_global_id(2);

            if (i >= Nx || j >= Ny || k >= Nz) return;

            int idx = i * Ny * Nz + j * Nz + k;

            // Backward differences with boundary conditions (matching _bwd axis zero padding)
            float dHz_dy = (j > 0) ? (Hz[idx] - Hz[idx - Nz]) : 0.0f;
            float dHy_dz = (k > 0) ? (Hy[idx] - Hy[idx - 1])  : 0.0f;
            float dHx_dz = (k > 0) ? (Hx[idx] - Hx[idx - 1])  : 0.0f;
            float dHz_dx = (i > 0) ? (Hz[idx] - Hz[idx - Ny * Nz]) : 0.0f;
            float dHy_dx = (i > 0) ? (Hy[idx] - Hy[idx - Ny * Nz]) : 0.0f;
            float dHx_dy = (j > 0) ? (Hx[idx] - Hx[idx - Nz]) : 0.0f;

            // CPML psi updates
            float p_Ex_y = by[j] * psi_Ex_y[idx] + cy[j] * dHz_dy;
            float p_Ex_z = bz[k] * psi_Ex_z[idx] + cz[k] * dHy_dz;
            float p_Ey_x = bx[i] * psi_Ey_x[idx] + cx[i] * dHz_dx;
            float p_Ey_z = bz[k] * psi_Ey_z[idx] + cz[k] * dHx_dz;
            float p_Ez_x = bx[i] * psi_Ez_x[idx] + cx[i] * dHy_dx;
            float p_Ez_y = by[j] * psi_Ez_y[idx] + cy[j] * dHx_dy;

            psi_Ex_y[idx] = p_Ex_y;
            psi_Ex_z[idx] = p_Ex_z;
            psi_Ey_x[idx] = p_Ey_x;
            psi_Ey_z[idx] = p_Ey_z;
            psi_Ez_x[idx] = p_Ez_x;
            psi_Ez_y[idx] = p_Ez_y;

            // Field update
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
            int i = get_global_id(0);
            int j = get_global_id(1);

            if (i >= Nx || j >= Ny) return;

            int idx = i * Ny * Nz + j * Nz + z_src;
            Ex[idx] += amp;
        }

        __kernel void accumulate_dft(
            int Nx, int Ny, int Nz,
            int ix0, int ix1,
            int iy0, int iy1,
            int iz0, int iz1,
            float phase_real, float phase_imag,
            __global const float *field,
            __global float2 *field_dft
        ) {
            int i = get_global_id(0);
            int j = get_global_id(1);
            int k = get_global_id(2);

            int x_dim = ix1 - ix0 + 1;
            int y_dim = iy1 - iy0 + 1;
            int z_dim = iz1 - iz0 + 1;

            if (i >= x_dim || j >= y_dim || k >= z_dim) return;

            int abs_i = ix0 + i;
            int abs_j = iy0 + j;
            int abs_k = iz0 + k;

            // Only update if it is on one of the 6 box faces
            if (abs_i == ix0 || abs_i == ix1 || abs_j == iy0 || abs_j == iy1 || abs_k == iz0 || abs_k == iz1) {
                int idx = abs_i * Ny * Nz + abs_j * Nz + abs_k;
                float val = field[idx];

                // DFT complex accumulation
                float2 current_dft = field_dft[idx];
                current_dft.x += val * phase_real;
                current_dft.y += val * phase_imag;
                field_dft[idx] = current_dft;
            }
        }
        """
        self.program = cl.Program(self.ctx, kernel_src).build()
        self.kern_update_H = cl.Kernel(self.program, "update_H")
        self.kern_update_E = cl.Kernel(self.program, "update_E")
        self.kern_add_source_Ex = cl.Kernel(self.program, "add_source_Ex")
        self.kern_accumulate_dft = cl.Kernel(self.program, "accumulate_dft")

    def add_source_Ex(self, z_src, amp):
        """Adds a sheet source value directly on the GPU using a kernel."""
        self.kern_add_source_Ex(
            self.queue,
            (self.Nx, self.Ny),
            None,
            np.int32(self.Nx), np.int32(self.Ny), np.int32(self.Nz),
            np.int32(z_src), np.float32(amp),
            self.Ex_buf
        )

    def _update_H(self):
        dtm = self.dt / MU0
        self.kern_update_H(
            self.queue,
            (self.Nx, self.Ny, self.Nz),
            None,
            np.int32(self.Nx), np.int32(self.Ny), np.int32(self.Nz),
            np.float32(self.dl), np.float32(dtm),
            self.Ex_buf, self.Ey_buf, self.Ez_buf,
            self.Hx_buf, self.Hy_buf, self.Hz_buf,
            self.bx_buf, self.cx_buf, self.kx_buf,
            self.by_buf, self.cy_buf, self.ky_buf,
            self.bz_buf, self.cz_buf, self.kz_buf,
            self.psi_Hx_y_buf, self.psi_Hx_z_buf,
            self.psi_Hy_x_buf, self.psi_Hy_z_buf,
            self.psi_Hz_x_buf, self.psi_Hz_y_buf
        )

    def _update_E(self):
        self.kern_update_E(
            self.queue,
            (self.Nx, self.Ny, self.Nz),
            None,
            np.int32(self.Nx), np.int32(self.Ny), np.int32(self.Nz),
            np.float32(self.dl), np.float32(self.dt),
            np.float32(EPS0),
            self.eps_buf,
            self.Hx_buf, self.Hy_buf, self.Hz_buf,
            self.Ex_buf, self.Ey_buf, self.Ez_buf,
            self.bx_buf, self.cx_buf, self.kx_buf,
            self.by_buf, self.cy_buf, self.ky_buf,
            self.bz_buf, self.cz_buf, self.kz_buf,
            self.psi_Ex_y_buf, self.psi_Ex_z_buf,
            self.psi_Ey_x_buf, self.psi_Ey_z_buf,
            self.psi_Ez_x_buf, self.psi_Ez_y_buf
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
