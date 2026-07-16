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

"""CUDA engine tests.

The FP32 CUDA engine is required to agree with the OpenCL engine (same
kernels, same arithmetic order); this is the contract enforced here whenever
a CUDA device is present. FP64 is validated against the float64 NumPy
reference engine.
"""

import unittest

import numpy as np
from opencl_fdtd_solver import NumPyFDTD, OpenCLFDTD

try:
    import cupy as cp

    try:
        _HAVE_CUDA = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # driver missing / no device
        _HAVE_CUDA = False
except ImportError:
    _HAVE_CUDA = False

if _HAVE_CUDA:
    from opencl_fdtd_solver import (
        CUDAFDTD,
        CUDANear2FarMonitor,
        NumPyNear2FarMonitor,
        OpenCLNear2FarMonitor,
    )

FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _sphere_eps(shape, ctr=15, rad=6, hi=4.0):
    eps = np.ones(shape, dtype=np.float32)
    i, j, k = np.indices(shape)
    eps[(i - ctr) ** 2 + (j - ctr) ** 2 + (k - ctr) ** 2 < rad**2] = hi
    return eps


@unittest.skipUnless(_HAVE_CUDA, "CUDA device (and cupy) required")
class TestCUDASolver(unittest.TestCase):
    def setUp(self):
        self.shape = (30, 30, 30)
        self.dl = 2e-3  # 2 mm
        self.npml = 6
        self.freq = 5e9  # 5 GHz
        self.eps = _sphere_eps(self.shape)
        self.z_src = self.shape[2] - self.npml - 2

    def _make(self, cls, dtype=np.float32, rim_taper=False):
        sim = cls(self.shape, self.dl, npml=self.npml, dtype=dtype)
        sim.set_epsilon(self.eps)
        sim.add_source(
            lambda f: f.add_source_Jx(
                self.z_src, np.sin(2 * np.pi * self.freq * f.t), rim_taper=rim_taper
            )
        )
        return sim

    def test_cuda_agrees_with_opencl_fp32(self):
        """Enforced contract: FP32 CUDA fields match the OpenCL engine."""
        cu = self._make(CUDAFDTD)
        cl = self._make(OpenCLFDTD)

        for _ in range(50):
            cu.step()
            cl.step()

        for field in FIELDS:
            cu_f = getattr(cu, field)
            cl_f = getattr(cl, field)
            self.assertGreater(np.max(np.abs(cl_f)), 0.0, f"OpenCL field {field} is zero!")
            diff = np.max(np.abs(cu_f - cl_f))
            self.assertLess(diff, 1e-4, f"CUDA/OpenCL mismatch in {field}: {diff:.6e}")

    def test_cuda_agrees_with_opencl_fp32_rim_taper(self):
        """Jx rim taper path matches OpenCL as well."""
        cu = self._make(CUDAFDTD, rim_taper=True)
        cl = self._make(OpenCLFDTD, rim_taper=True)
        for _ in range(30):
            cu.step()
            cl.step()
        for field in FIELDS:
            diff = np.max(np.abs(getattr(cu, field) - getattr(cl, field)))
            self.assertLess(diff, 1e-4, f"CUDA/OpenCL mismatch in {field}: {diff:.6e}")

    def test_cuda_fp64_matches_numpy_fp64(self):
        """FP64 CUDA fields track the float64 NumPy reference tightly."""
        cu = self._make(CUDAFDTD, dtype=np.float64)
        ref = self._make(NumPyFDTD, dtype=np.float64)

        for _ in range(50):
            cu.step()
            ref.step()

        for field in FIELDS:
            ref_f = getattr(ref, field)
            self.assertGreater(np.max(np.abs(ref_f)), 0.0, f"NumPy field {field} is zero!")
            diff = np.max(np.abs(getattr(cu, field) - ref_f))
            self.assertLess(diff, 1e-9, f"CUDA FP64/NumPy mismatch in {field}: {diff:.6e}")

    def test_cuda_fp32_fp64_consistent(self):
        """FP32 and FP64 CUDA runs describe the same physics (loose bound)."""
        cu32 = self._make(CUDAFDTD, dtype=np.float32)
        cu64 = self._make(CUDAFDTD, dtype=np.float64)
        for _ in range(50):
            cu32.step()
            cu64.step()
        for field in ("Ex", "Ez"):
            f32 = getattr(cu32, field).astype(np.float64)
            f64 = getattr(cu64, field)
            scale = np.max(np.abs(f64))
            self.assertGreater(scale, 0.0)
            rel = np.max(np.abs(f32 - f64)) / scale
            self.assertLess(rel, 1e-3, f"FP32/FP64 divergence in {field}: {rel:.6e}")

    def test_cuda_rejects_unsupported_dtype(self):
        with self.assertRaises(ValueError):
            CUDAFDTD(self.shape, self.dl, npml=self.npml, dtype=np.float16)

    def test_cuda_monitors_agree_with_opencl(self):
        """CUDA face-DFT + far-field agree with the OpenCL monitor and the
        host NumPy monitor (FP32, enforced)."""
        cu = self._make(CUDAFDTD)
        cl = self._make(OpenCLFDTD)

        ctr_phys = (30e-3, 30e-3, 30e-3)
        size_phys = (20e-3, 20e-3, 20e-3)

        cu_mon = CUDANear2FarMonitor(cu, ctr_phys, size_phys, self.freq)
        np_mon = NumPyNear2FarMonitor(cu, ctr_phys, size_phys, self.freq)
        cl_mon = OpenCLNear2FarMonitor(cl, ctr_phys, size_phys, self.freq)

        for _ in range(80):
            cu.step()
            cl.step()

        obs_list = [
            (0.0, 0.0, 1000.0),
            (1000.0, 0.0, 0.0),
            (707.1, 0.0, 707.1),
        ]
        ff_cu = cu_mon.get_farfields(obs_list)
        ff_cl = cl_mon.get_farfields(obs_list)
        scale = np.max(np.abs(ff_cl))
        self.assertGreater(scale, 0.0, "OpenCL far-field is zero")
        rel = np.max(np.abs(ff_cu - ff_cl)) / scale
        self.assertLess(rel, 1e-4, f"CUDA/OpenCL far-field mismatch: {rel:.6e}")

        # Host NumPy monitor sums the DFT in a different order (complex64
        # accumulation), so this cross-check is looser than the CUDA/OpenCL
        # contract above.
        for obs in obs_list:
            ff_np = np_mon.get_farfield(obs)
            diff = np.max(np.abs(cu_mon.get_farfield(obs) - ff_np))
            self.assertLess(diff / scale, 5e-3, f"CUDA/NumPy far-field mismatch at {obs}")

        # Face-only accumulation stays much smaller than the full volume.
        n_cells = int(np.prod(self.shape))
        self.assertLess(cu_mon.n_face_samples, n_cells // 2)

        # Polar helper returns finite dB samples.
        ang, db = cu_mon.farfield_polar_xz(distance_m=1000.0, n_angles=9)
        self.assertEqual(len(ang), 9)
        self.assertTrue(np.all(np.isfinite(db)))

        # Sparse volume scatter for debugging.
        cu_mon.fetch_dft_fields()
        self.assertGreater(np.max(np.abs(cu_mon.Ex_dft)), 0.0)

    def test_cuda_dft_relative_change(self):
        """Device-side DFT convergence metric behaves like the OpenCL one."""
        cu = self._make(CUDAFDTD)
        mon = CUDANear2FarMonitor(cu, (30e-3, 30e-3, 30e-3), (20e-3, 20e-3, 20e-3), self.freq)

        # No snapshot yet -> not converged.
        self.assertEqual(mon.dft_relative_change(), 1.0)

        mon.snapshot_dft()
        cu.run(40)
        first = mon.dft_relative_change()
        self.assertTrue(0.0 < first <= 1.0)

        mon.snapshot_dft()
        cu.run(40)
        second = mon.dft_relative_change()
        self.assertTrue(0.0 <= second < 1.0)
        self.assertLess(second, first)

    def test_cuda_fp64_monitor_runs(self):
        """FP64 monitor pipeline (double2 DFT + double atomics) produces
        far fields consistent with the FP32 path."""
        cu64 = self._make(CUDAFDTD, dtype=np.float64)
        cu32 = self._make(CUDAFDTD, dtype=np.float32)
        args = ((30e-3, 30e-3, 30e-3), (20e-3, 20e-3, 20e-3), self.freq)
        m64 = CUDANear2FarMonitor(cu64, *args)
        m32 = CUDANear2FarMonitor(cu32, *args)
        for _ in range(80):
            cu64.step()
            cu32.step()
        obs = [(0.0, 0.0, 1000.0), (707.1, 0.0, 707.1)]
        ff64 = m64.get_farfields(obs)
        ff32 = m32.get_farfields(obs)
        scale = np.max(np.abs(ff64))
        self.assertGreater(scale, 0.0)
        self.assertLess(np.max(np.abs(ff64 - ff32)) / scale, 1e-2)


if __name__ == "__main__":
    unittest.main()
