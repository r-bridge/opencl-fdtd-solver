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

from .constants import C0, EPS0, ETA0, MU0
from .kernels import load_kernel_source
from .plugin import SourceMonitorMixin

# Reuse one context/queue for default OpenCLFDTD construction. Creating a new
# cl.Context per instance is fine on discrete GPUs but can segfault POCL (CI)
# when many contexts are destroyed at process exit.
_DEFAULT_CTX = None
_DEFAULT_QUEUE = None
_DEFAULT_DEVICE = None


def _default_opencl_runtime():
    """Return a process-wide (context, queue, device) for OpenCLFDTD defaults."""
    global _DEFAULT_CTX, _DEFAULT_QUEUE, _DEFAULT_DEVICE
    if _DEFAULT_CTX is not None:
        return _DEFAULT_CTX, _DEFAULT_QUEUE, _DEFAULT_DEVICE

    platforms = cl.get_platforms()
    if not platforms:
        raise RuntimeError("No OpenCL platforms found.")
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

    _DEFAULT_DEVICE = devices[0]
    _DEFAULT_CTX = cl.Context([_DEFAULT_DEVICE])
    _DEFAULT_QUEUE = cl.CommandQueue(_DEFAULT_CTX)
    return _DEFAULT_CTX, _DEFAULT_QUEUE, _DEFAULT_DEVICE


class OpenCLFDTD(SourceMonitorMixin):
    """
    3D Yee-grid FDTD electromagnetic solver accelerated with OpenCL.

    Accepts 3D epsilon array, compiles OpenCL update kernels, and runs the simulation loop.
    Supports pluggable monitors (NumPy and OpenCL models).
    """

    # Leave headroom for the OpenCL runtime, framebuffer, and other processes.
    # Without this, allocation may "succeed" while the driver pages to host RAM
    # and effective throughput collapses by an order of magnitude.
    MEMORY_HEADROOM_FRACTION = 0.12
    MEMORY_HEADROOM_BYTES = 512 * 1024 * 1024

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
            self.ctx, shared_queue, self.device = _default_opencl_runtime()
            if queue is None:
                queue = shared_queue
        else:
            self.ctx = ctx
            self.device = self.ctx.devices[0]

        if queue is None:
            self.queue = cl.CommandQueue(self.ctx)
        else:
            self.queue = queue

        print(f"OpenCL FDTD Solver initialized on device: {self.device.name}")

        self._check_device_memory(shape, self.npml, self.dtype)

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
        dl = self.dl
        dt = self.dt
        npml = self.npml
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz

        m = 3
        sigma_opt = 0.8 * (m + 1) / (2.0 * ETA0 * dl * npml) if npml > 0 else 0.0
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

    @classmethod
    def device_memory_budget_bytes(cls, device):
        """Usable device memory after reserved headroom."""
        total = int(device.global_mem_size)
        reserve = max(
            int(total * cls.MEMORY_HEADROOM_FRACTION),
            int(cls.MEMORY_HEADROOM_BYTES),
        )
        return max(0, total - reserve)

    def _check_device_memory(self, shape, npml, dtype):
        """Raise before allocation if the model cannot fit with headroom."""
        needed = self.estimate_device_memory_bytes(shape, npml, dtype)
        budget = self.device_memory_budget_bytes(self.device)
        total = int(self.device.global_mem_size)
        if needed > budget:
            reserve = max(
                int(total * self.MEMORY_HEADROOM_FRACTION),
                int(self.MEMORY_HEADROOM_BYTES),
            )
            raise MemoryError(
                f"Model needs ~{needed / (1024**3):.2f} GB device memory, but "
                f"{self.device.name} only has ~{budget / (1024**3):.2f} GB usable "
                f"({total / (1024**3):.2f} GB total minus "
                f"{reserve / (1024**3):.2f} GB headroom). "
                f"Reduce the grid or npml; continuing would risk silent host paging "
                f"and order-of-magnitude slower runs."
            )

    def _compile_kernels(self):
        """Compile Yee-grid FDTD update kernels (coalesced NDRange + interior/PML split)."""
        # Work-item mapping: get_global_id(0)=k, (1)=j, (2)=i so adjacent threads
        # touch contiguous addresses along the fastest array axis (k).
        # Kernel sources live in opencl_fdtd_solver/kernels/*.cl
        kernel_src = load_kernel_source()
        self.program = cl.Program(self.ctx, kernel_src).build()
        self.kern_update_H_interior = cl.Kernel(self.program, "update_H_interior")
        self.kern_update_H_pml = cl.Kernel(self.program, "update_H_pml")
        self.kern_update_E_interior = cl.Kernel(self.program, "update_E_interior")
        self.kern_update_E_pml = cl.Kernel(self.program, "update_E_pml")
        self.kern_add_source_Ex = cl.Kernel(self.program, "add_source_Ex")
        self.kern_add_source_Jx = cl.Kernel(self.program, "add_source_Jx")
        self.kern_accumulate_dft = cl.Kernel(self.program, "accumulate_dft")
        self.kern_accumulate_dft_face = cl.Kernel(self.program, "accumulate_dft_face")
        self.kern_accumulate_dft_faces_fused = cl.Kernel(self.program, "accumulate_dft_faces_fused")
        self.kern_dft_rel_change_partial = cl.Kernel(self.program, "dft_rel_change_partial")
        self.kern_farfield_accumulate_nl = cl.Kernel(self.program, "farfield_accumulate_nl")
        self.kern_farfield_nl_to_eh = cl.Kernel(self.program, "farfield_nl_to_eh")

        # Cached launch geometries (coalesced: Nz, Ny, Nx).
        self._gs_full = (self.Nz, self.Ny, self.Nx)
        n = self.npml
        nx_i = self.Nx - 2 * n
        ny_i = self.Ny - 2 * n
        nz_i = self.Nz - 2 * n
        self._gs_interior = (nz_i, ny_i, nx_i) if (nx_i > 0 and ny_i > 0 and nz_i > 0) else None

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
        self.kern_add_source_Ex(
            self.queue,
            (self.Ny, self.Nx),
            None,
            np.int32(self.Nx),
            np.int32(self.Ny),
            np.int32(self.Nz),
            np.int32(z_src),
            np.float32(amp),
            np.int32(i0_i),
            np.int32(i1_i),
            np.int32(j0_i),
            np.int32(j1_i),
            self.Ex_buf,
        )

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

        Applies ``Ex += -dt/(ε₀ εᵣ) Jx`` using the on-device ε buffer, matching
        Meep's ``D -= J·dt`` then ``E = χ⁻¹ D`` (with SI ε₀ restored).

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
        jx = float(Jx)
        re = float(rim_edge)
        if rim_taper and rim_renorm:
            nx_s = max(0, i1_i - i0_i)
            ny_s = max(0, j1_i - j0_i)
            if nx_s >= 2 and ny_s >= 2:
                ni, nj = nx_s - 2, ny_s - 2
                wsum = ni * nj + re * (2 * ni + 2 * nj) + (re * re) * 4
                jx *= (nx_s * ny_s) / wsum
        self.kern_add_source_Jx(
            self.queue,
            (self.Ny, self.Nx),
            None,
            np.int32(self.Nx),
            np.int32(self.Ny),
            np.int32(self.Nz),
            np.int32(z_src),
            np.float32(jx),
            np.float32(self.dt),
            np.float32(EPS0),
            np.int32(i0_i),
            np.int32(i1_i),
            np.int32(j0_i),
            np.int32(j1_i),
            np.int32(1 if rim_taper else 0),
            np.float32(re),
            self.eps_buf,
            self.Ex_buf,
        )

    def _update_H(self):
        dtm = self.dt / MU0
        nx, ny, nz = np.int32(self.Nx), np.int32(self.Ny), np.int32(self.Nz)
        npml = np.int32(self.npml)
        dl = np.float32(self.dl)
        dtm_f = np.float32(dtm)

        if self.npml > 0:
            if self._gs_interior is not None:
                self.kern_update_H_interior(
                    self.queue,
                    self._gs_interior,
                    None,
                    nx,
                    ny,
                    nz,
                    npml,
                    dl,
                    dtm_f,
                    self.Ex_buf,
                    self.Ey_buf,
                    self.Ez_buf,
                    self.Hx_buf,
                    self.Hy_buf,
                    self.Hz_buf,
                )
            self.kern_update_H_pml(
                self.queue,
                self._gs_full,
                None,
                nx,
                ny,
                nz,
                npml,
                dl,
                dtm_f,
                self.Ex_buf,
                self.Ey_buf,
                self.Ez_buf,
                self.Hx_buf,
                self.Hy_buf,
                self.Hz_buf,
                self.bx_buf,
                self.cx_buf,
                self.kx_buf,
                self.by_buf,
                self.cy_buf,
                self.ky_buf,
                self.bz_buf,
                self.cz_buf,
                self.kz_buf,
                self.psi_Hx_y_buf,
                self.psi_Hx_z_buf,
                self.psi_Hy_x_buf,
                self.psi_Hy_z_buf,
                self.psi_Hz_x_buf,
                self.psi_Hz_y_buf,
            )
        else:
            # npml==0: use boundary-guarded full-domain kernel. The psi-free
            # interior kernel assumes a viable stencil neighborhood and will
            # read out-of-bounds if launched over the entire grid.
            self.kern_update_H_pml(
                self.queue,
                self._gs_full,
                None,
                nx,
                ny,
                nz,
                np.int32(0),
                dl,
                dtm_f,
                self.Ex_buf,
                self.Ey_buf,
                self.Ez_buf,
                self.Hx_buf,
                self.Hy_buf,
                self.Hz_buf,
                self.bx_buf,
                self.cx_buf,
                self.kx_buf,
                self.by_buf,
                self.cy_buf,
                self.ky_buf,
                self.bz_buf,
                self.cz_buf,
                self.kz_buf,
                self.psi_Hx_y_buf,
                self.psi_Hx_z_buf,
                self.psi_Hy_x_buf,
                self.psi_Hy_z_buf,
                self.psi_Hz_x_buf,
                self.psi_Hz_y_buf,
            )

    def _update_E(self):
        nx, ny, nz = np.int32(self.Nx), np.int32(self.Ny), np.int32(self.Nz)
        npml = np.int32(self.npml)
        dl = np.float32(self.dl)
        dt = np.float32(self.dt)
        eps0 = np.float32(EPS0)

        if self.npml > 0:
            if self._gs_interior is not None:
                self.kern_update_E_interior(
                    self.queue,
                    self._gs_interior,
                    None,
                    nx,
                    ny,
                    nz,
                    npml,
                    dl,
                    dt,
                    eps0,
                    self.eps_buf,
                    self.Hx_buf,
                    self.Hy_buf,
                    self.Hz_buf,
                    self.Ex_buf,
                    self.Ey_buf,
                    self.Ez_buf,
                )
            self.kern_update_E_pml(
                self.queue,
                self._gs_full,
                None,
                nx,
                ny,
                nz,
                npml,
                dl,
                dt,
                eps0,
                self.eps_buf,
                self.Hx_buf,
                self.Hy_buf,
                self.Hz_buf,
                self.Ex_buf,
                self.Ey_buf,
                self.Ez_buf,
                self.bx_buf,
                self.cx_buf,
                self.kx_buf,
                self.by_buf,
                self.cy_buf,
                self.ky_buf,
                self.bz_buf,
                self.cz_buf,
                self.kz_buf,
                self.psi_Ex_y_buf,
                self.psi_Ex_z_buf,
                self.psi_Ey_x_buf,
                self.psi_Ey_z_buf,
                self.psi_Ez_x_buf,
                self.psi_Ez_y_buf,
            )
        else:
            self.kern_update_E_pml(
                self.queue,
                self._gs_full,
                None,
                nx,
                ny,
                nz,
                np.int32(0),
                dl,
                dt,
                eps0,
                self.eps_buf,
                self.Hx_buf,
                self.Hy_buf,
                self.Hz_buf,
                self.Ex_buf,
                self.Ey_buf,
                self.Ez_buf,
                self.bx_buf,
                self.cx_buf,
                self.kx_buf,
                self.by_buf,
                self.cy_buf,
                self.ky_buf,
                self.bz_buf,
                self.cz_buf,
                self.kz_buf,
                self.psi_Ex_y_buf,
                self.psi_Ex_z_buf,
                self.psi_Ey_x_buf,
                self.psi_Ey_z_buf,
                self.psi_Ez_x_buf,
                self.psi_Ez_y_buf,
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
        self.queue.finish()
        return host_arr

    def read_point(self, field_name, i, j, k):
        """Read a single point value from a field buffer on the GPU (efficient 4-byte copy)."""
        buf = getattr(self, f"{field_name}_buf")
        idx = int(i * self.Ny * self.Nz + j * self.Nz + k)
        dest = np.empty(1, dtype=self.dtype)
        cl.enqueue_copy(self.queue, dest, buf, src_offset=idx * 4)
        self.queue.finish()
        return float(dest[0])

    @property
    def Ex(self):
        return self._read_field(self.Ex_buf)

    @property
    def Ey(self):
        return self._read_field(self.Ey_buf)

    @property
    def Ez(self):
        return self._read_field(self.Ez_buf)

    @property
    def Hx(self):
        return self._read_field(self.Hx_buf)

    @property
    def Hy(self):
        return self._read_field(self.Hy_buf)

    @property
    def Hz(self):
        return self._read_field(self.Hz_buf)
