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

"""CUDA FDTD engine (CuPy/NVRTC), numerically matching :class:`OpenCLFDTD`.

Supports float32 and float64 computation from one templated kernel source.
The float32 path keeps the same arithmetic order as the OpenCL kernels so the
two engines agree within roundoff; this is enforced by
``tests/test_cuda_solver.py``.
"""

import logging
import warnings

import cupy as cp
import numpy as np

from .constants import C0, EPS0, MU0
from .cpml import build_cpml_profiles
from .kernels import load_cuda_kernel_source
from .materials import yee_edge_ce
from .plugin import SourceMonitorMixin

_KERNEL_NAMES = (
    "update_H_interior",
    "update_H_pml",
    "update_E_interior",
    "update_E_pml",
    "add_source_Ex",
    "add_source_Jx",
    "accumulate_dft",
    "accumulate_dft_face",
    "accumulate_dft_faces_fused",
    "dft_rel_change_partial",
    "farfield_accumulate_nl",
    "farfield_nl_to_eh",
)

# One compiled module per (device, dtype); NVRTC compilation is not free.
_MODULE_CACHE: dict[tuple[int, str], cp.RawModule] = {}


def _get_module(device_id: int, dtype: np.dtype) -> cp.RawModule:
    key = (device_id, dtype.name)
    mod = _MODULE_CACHE.get(key)
    if mod is None:
        t = "float" if dtype == np.float32 else "double"
        # NVRTC defaults (fmad on, no fast-math) mirror the OpenCL build's
        # -cl-mad-enable without -cl-fast-relaxed-math.
        mod = cp.RawModule(
            code=load_cuda_kernel_source(),
            options=(
                f"-Dreal={t}",
                f"-Dreal2={t}2",
                f"-Dmake_real2=make_{t}2",
            ),
            name_expressions=None,
        )
        _MODULE_CACHE[key] = mod
    return mod


class CUDAFDTD(SourceMonitorMixin):
    """
    3D Yee-grid FDTD electromagnetic solver accelerated with CUDA (CuPy).

    Accepts a 3D epsilon array, compiles CUDA update kernels via NVRTC, and
    runs the simulation loop. Supports pluggable monitors (same protocol as
    :class:`OpenCLFDTD`). Computation dtype is ``np.float32`` or ``np.float64``.
    """

    # Same policy as OpenCLFDTD: leave headroom for the runtime/framebuffer.
    MEMORY_HEADROOM_FRACTION = 0.12
    MEMORY_HEADROOM_BYTES = 512 * 1024 * 1024

    def __init__(self, shape, dl, npml=20, dtype=np.float32, device=None):
        """
        shape  : (Nx, Ny, Nz) Yee cells
        dl     : uniform cell size in metres
        npml   : PML thickness in cells
        dtype  : computation dtype (``np.float32`` or ``np.float64``)
        device : CUDA device ordinal (optional; default: current device)
        """
        self.Nx, self.Ny, self.Nz = shape
        self.dl = float(dl)
        self.npml = int(npml)
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError(f"CUDAFDTD supports float32 or float64 computation; got {dtype!r}")
        self.dtype = dtype
        self.complex_dtype = np.dtype(np.complex64 if dtype == np.float32 else np.complex128)
        # Scalar constructor matching the kernels' ``real`` type.
        self.real = np.float32 if dtype == np.float32 else np.float64
        self.t = 0.0
        self.step_num = 0

        # Courant-stable time step
        self.dt = 0.99 * dl / (C0 * np.sqrt(3.0))

        self.device = cp.cuda.Device() if device is None else cp.cuda.Device(int(device))
        self.device.use()
        props = cp.cuda.runtime.getDeviceProperties(self.device.id)
        self.device_name = props["name"].decode(errors="replace")
        logging.getLogger(__name__).info(
            "CUDA FDTD Solver initialized on device: %s", self.device_name
        )

        self._check_device_memory(shape, self.npml, self.dtype)

        # Yee field arrays size
        self.size = self.Nx * self.Ny * self.Nz

        # Allocate Yee fields on GPU (fields=0, eps_r=1)
        self.Ex_buf = cp.zeros(self.size, dtype=self.dtype)
        self.Ey_buf = cp.zeros(self.size, dtype=self.dtype)
        self.Ez_buf = cp.zeros(self.size, dtype=self.dtype)
        self.Hx_buf = cp.zeros(self.size, dtype=self.dtype)
        self.Hy_buf = cp.zeros(self.size, dtype=self.dtype)
        self.Hz_buf = cp.zeros(self.size, dtype=self.dtype)
        # Per-component E-update coefficients dt/(eps0*eps_r) at Yee edges.
        ce_vac = self.dtype.type(self.dt / EPS0)
        self.ce_x_buf = cp.full(self.size, ce_vac, dtype=self.dtype)
        self.ce_y_buf = cp.full(self.size, ce_vac, dtype=self.dtype)
        self.ce_z_buf = cp.full(self.size, ce_vac, dtype=self.dtype)
        # Alias: Ex / Jx sources use the Ex-edge coefficient.
        self.ce_buf = self.ce_x_buf

        self._sources = []
        self._monitors = []

        self._build_cpml()
        self._compile_kernels()

    def set_epsilon(self, eps_array):
        """Set cell-centered ``εᵣ``; store Yee-edge ``ce = dt/(ε₀ εᵣ)`` on the GPU."""
        expected = (self.Nx, self.Ny, self.Nz)
        if eps_array.shape != expected:
            raise ValueError(
                f"Epsilon shape mismatch: expected {expected}, got {tuple(eps_array.shape)}"
            )
        ce_x, ce_y, ce_z = yee_edge_ce(eps_array, self.dt, dtype=self.dtype)
        self.ce_x_buf.set(ce_x.ravel())
        self.ce_y_buf.set(ce_y.ravel())
        self.ce_z_buf.set(ce_z.ravel())

    def _build_cpml(self):
        """Calculate CPML coefficients and allocate face-local psi buffers on the GPU."""
        dl = self.dl
        npml = self.npml
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz

        profiles = build_cpml_profiles((Nx, Ny, Nz), npml=npml, dl=dl, dt=self.dt, dtype=self.dtype)

        def _ik(kappa: np.ndarray) -> np.ndarray:
            # Kernels multiply by 1/(kappa*dl) instead of dividing by kappa*dl.
            return (1.0 / (kappa.astype(np.float64) * dl)).astype(self.dtype)

        def _upload_axis(prof):
            return (
                cp.asarray(prof.b),
                cp.asarray(prof.c),
                cp.asarray(_ik(prof.kappa)),
            )

        # H-update uses H-node stagger; E-update uses E-node stagger.
        self.bx_h_buf, self.cx_h_buf, self.kx_h_buf = _upload_axis(profiles.h[0])
        self.by_h_buf, self.cy_h_buf, self.ky_h_buf = _upload_axis(profiles.h[1])
        self.bz_h_buf, self.cz_h_buf, self.kz_h_buf = _upload_axis(profiles.h[2])
        self.bx_e_buf, self.cx_e_buf, self.kx_e_buf = _upload_axis(profiles.e[0])
        self.by_e_buf, self.cy_e_buf, self.ky_e_buf = _upload_axis(profiles.e[1])
        self.bz_e_buf, self.cz_e_buf, self.kz_e_buf = _upload_axis(profiles.e[2])

        # Back-compat aliases (H-node set), mirroring OpenCLFDTD.
        self.bx_buf, self.cx_buf, self.kx_buf = self.bx_h_buf, self.cx_h_buf, self.kx_h_buf
        self.by_buf, self.cy_buf, self.ky_buf = self.by_h_buf, self.cy_h_buf, self.ky_h_buf
        self.bz_buf, self.cz_buf, self.kz_buf = self.bz_h_buf, self.cz_h_buf, self.kz_h_buf

        # Face-local psi: only the PML slabs where the corresponding c-coeff is nonzero.
        self.psi_x_size = (2 * npml * Ny * Nz) if npml > 0 else 0
        self.psi_y_size = (Nx * 2 * npml * Nz) if npml > 0 else 0
        self.psi_z_size = (Nx * Ny * 2 * npml) if npml > 0 else 0

        def _psi_buf(n):
            # Tiny placeholder so kernel args remain valid if ever referenced.
            return cp.zeros(max(n, 1), dtype=self.dtype)

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
        fields = 9 * nx * ny * nz * item  # Ex..Hz + ce_x/y/z (Yee-edge coeffs)
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
        with device:
            total = int(device.mem_info[1])
        reserve = max(
            int(total * cls.MEMORY_HEADROOM_FRACTION),
            int(cls.MEMORY_HEADROOM_BYTES),
        )
        return max(0, total - reserve)

    def _check_device_memory(self, shape, npml, dtype):
        """Raise before allocation if the model cannot fit with headroom."""
        needed = self.estimate_device_memory_bytes(shape, npml, dtype)
        budget = self.device_memory_budget_bytes(self.device)
        total = int(self.device.mem_info[1])
        if needed > budget:
            reserve = max(
                int(total * self.MEMORY_HEADROOM_FRACTION),
                int(self.MEMORY_HEADROOM_BYTES),
            )
            raise MemoryError(
                f"Model needs ~{needed / (1024**3):.2f} GB device memory, but "
                f"{self.device_name if hasattr(self, 'device_name') else 'CUDA device'} "
                f"only has ~{budget / (1024**3):.2f} GB usable "
                f"({total / (1024**3):.2f} GB total minus "
                f"{reserve / (1024**3):.2f} GB headroom). "
                f"Reduce the grid or npml; continuing would risk silent host paging "
                f"and order-of-magnitude slower runs."
            )

    def _compile_kernels(self):
        """Compile Yee-grid FDTD update kernels (coalesced grid + interior/PML split)."""
        # Thread mapping: x=k, y=j, z=i so adjacent threads touch contiguous
        # addresses along the fastest array axis (k), same as the OpenCL build.
        module = _get_module(self.device.id, self.dtype)
        for name in _KERNEL_NAMES:
            setattr(self, f"kern_{name}", module.get_function(name))

        # Explicit block size: a warp-multiple along the fastest axis k.
        # Grids are rounded up; kernels bounds-guard the padding threads.
        wg_cap = int(cp.cuda.runtime.getDeviceProperties(self.device.id)["maxThreadsPerBlock"])
        for name in ("update_H_interior", "update_H_pml", "update_E_interior", "update_E_pml"):
            wg_cap = min(wg_cap, int(getattr(self, f"kern_{name}").max_threads_per_block))
        lk = 128
        while lk > 1 and lk > wg_cap:
            lk //= 2
        self._lk = lk
        self._block_update = (lk, 1, 1)

        def _blocks(n_items):
            return (n_items + lk - 1) // lk

        self._grid_full = (_blocks(self.Nz), self.Ny, self.Nx)
        n = self.npml
        nx_i = self.Nx - 2 * n
        ny_i = self.Ny - 2 * n
        nz_i = self.Nz - 2 * n
        self._grid_interior = (
            (_blocks(nz_i), ny_i, nx_i) if (nx_i > 0 and ny_i > 0 and nz_i > 0) else None
        )

        # Sheet-source launch geometry: x=j (fastest), y=i.
        lj = min(lk, 128)
        self._block_source = (lj, 1, 1)
        self._grid_source = ((self.Ny + lj - 1) // lj, self.Nx, 1)

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
        i0_i = 0 if i0 is None else int(i0)
        i1_i = self.Nx if i1 is None else int(i1)
        j0_i = 0 if j0 is None else int(j0)
        j1_i = self.Ny if j1 is None else int(j1)
        self.kern_add_source_Ex(
            self._grid_source,
            self._block_source,
            (
                np.int32(self.Nx),
                np.int32(self.Ny),
                np.int32(self.Nz),
                np.int32(z_src),
                self.real(amp),
                np.int32(i0_i),
                np.int32(i1_i),
                np.int32(j0_i),
                np.int32(j1_i),
                self.Ex_buf,
            ),
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
        ``rim_edge``, corners × ``rim_edge²``). With ``rim_renorm`` (default
        true), ``Jx`` is scaled so ∑weights equals the hard cell count.
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
            self._grid_source,
            self._block_source,
            (
                np.int32(self.Nx),
                np.int32(self.Ny),
                np.int32(self.Nz),
                np.int32(z_src),
                self.real(jx),
                np.int32(i0_i),
                np.int32(i1_i),
                np.int32(j0_i),
                np.int32(j1_i),
                np.int32(1 if rim_taper else 0),
                self.real(re),
                self.ce_buf,
                self.Ex_buf,
            ),
        )

    def _update_H(self):
        nx, ny, nz = np.int32(self.Nx), np.int32(self.Ny), np.int32(self.Nz)
        npml = np.int32(self.npml)
        dtm_f = self.real(self.dt / MU0)
        dtm_dl = self.real(self.dt / (MU0 * self.dl))

        if self.npml > 0 and self._grid_interior is not None:
            self.kern_update_H_interior(
                self._grid_interior,
                self._block_update,
                (
                    nx,
                    ny,
                    nz,
                    npml,
                    dtm_dl,
                    self.Ex_buf,
                    self.Ey_buf,
                    self.Ez_buf,
                    self.Hx_buf,
                    self.Hy_buf,
                    self.Hz_buf,
                ),
            )
        # npml==0: the boundary-guarded kernel covers the whole grid (the
        # psi-free interior kernel would read out-of-bounds there).
        self.kern_update_H_pml(
            self._grid_full,
            self._block_update,
            (
                nx,
                ny,
                nz,
                npml,
                dtm_f,
                self.Ex_buf,
                self.Ey_buf,
                self.Ez_buf,
                self.Hx_buf,
                self.Hy_buf,
                self.Hz_buf,
                self.bx_h_buf,
                self.cx_h_buf,
                self.kx_h_buf,
                self.by_h_buf,
                self.cy_h_buf,
                self.ky_h_buf,
                self.bz_h_buf,
                self.cz_h_buf,
                self.kz_h_buf,
                self.psi_Hx_y_buf,
                self.psi_Hx_z_buf,
                self.psi_Hy_x_buf,
                self.psi_Hy_z_buf,
                self.psi_Hz_x_buf,
                self.psi_Hz_y_buf,
            ),
        )

    def _update_E(self):
        nx, ny, nz = np.int32(self.Nx), np.int32(self.Ny), np.int32(self.Nz)
        npml = np.int32(self.npml)
        inv_dl = self.real(1.0 / self.dl)
        ce = (self.ce_x_buf, self.ce_y_buf, self.ce_z_buf)

        if self.npml > 0 and self._grid_interior is not None:
            self.kern_update_E_interior(
                self._grid_interior,
                self._block_update,
                (
                    nx,
                    ny,
                    nz,
                    npml,
                    inv_dl,
                    *ce,
                    self.Hx_buf,
                    self.Hy_buf,
                    self.Hz_buf,
                    self.Ex_buf,
                    self.Ey_buf,
                    self.Ez_buf,
                ),
            )
        self.kern_update_E_pml(
            self._grid_full,
            self._block_update,
            (
                nx,
                ny,
                nz,
                npml,
                *ce,
                self.Hx_buf,
                self.Hy_buf,
                self.Hz_buf,
                self.Ex_buf,
                self.Ey_buf,
                self.Ez_buf,
                self.bx_e_buf,
                self.cx_e_buf,
                self.kx_e_buf,
                self.by_e_buf,
                self.cy_e_buf,
                self.ky_e_buf,
                self.bz_e_buf,
                self.cz_e_buf,
                self.kz_e_buf,
                self.psi_Ex_y_buf,
                self.psi_Ex_z_buf,
                self.psi_Ey_x_buf,
                self.psi_Ey_z_buf,
                self.psi_Ez_x_buf,
                self.psi_Ez_y_buf,
            ),
        )

    def step(self):
        """Single timestep Yee-update with sources and monitors."""
        self._update_H()
        # Soft currents belong at (n+1/2)Δt in the leapfrog E update.
        t_int = self.t
        self.t = t_int + 0.5 * self.dt
        for src in self._sources:
            src(self)
        self.t = t_int
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
                logging.getLogger(__name__).info("  step %d/%d  t=%.3e s", i, n_steps, self.t)

    # ── CPU Host properties to read Yee fields from GPU (NumPy compatibility) ──
    def _read_field(self, buf):
        return cp.asnumpy(buf).reshape(self.Nx, self.Ny, self.Nz)

    def read_point(self, field_name, i, j, k):
        """Read a single point value from a field buffer on the GPU."""
        buf = getattr(self, f"{field_name}_buf")
        idx = int(i * self.Ny * self.Nz + j * self.Nz + k)
        return float(buf[idx].get())

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
