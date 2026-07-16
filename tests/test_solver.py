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

import unittest

import numpy as np
from opencl_fdtd_solver import (
    NumPyFDTD,
    NumPyFDTD_FaceCPML,
    NumPyNear2FarMonitor,
    OpenCLFDTD,
    OpenCLNear2FarMonitor,
)


class TestGenericFDTDSolver(unittest.TestCase):
    def setUp(self):
        self.shape = (30, 30, 30)
        self.dl = 2e-3  # 2 mm
        self.npml = 6
        self.freq = 5e9  # 5 GHz

        # Build dummy permittivity profile (dielectric sphere at the center)
        self.eps = np.ones(self.shape, dtype=np.float32)
        ctr = 15
        rad = 6
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    if (i - ctr) ** 2 + (j - ctr) ** 2 + (k - ctr) ** 2 < rad**2:
                        self.eps[i, j, k] = 4.0

    def test_numpy_vs_opencl_engine_fields(self):
        """Verify that the OpenCL engine field updates numerically match the NumPy reference solver."""
        # Initialize both solvers
        np_sim = NumPyFDTD(self.shape, self.dl, npml=self.npml)
        cl_sim = OpenCLFDTD(self.shape, self.dl, npml=self.npml)

        np_sim.set_epsilon(self.eps)
        cl_sim.set_epsilon(self.eps)

        # Source parameters
        z_src = self.shape[2] - self.npml - 2

        def np_source(f):
            amp = np.sin(2 * np.pi * self.freq * f.t)
            f.Ex[:, :, z_src] += amp

        def cl_source(f):
            amp = np.sin(2 * np.pi * self.freq * f.t)
            f.add_source_Ex(z_src, amp)

        np_sim.add_source(np_source)
        cl_sim.add_source(cl_source)

        # Run both for 50 steps
        for _ in range(50):
            np_sim.step()
            cl_sim.step()

        # Check maximum difference across all field components
        for field in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
            np_field = getattr(np_sim, field)
            cl_field = getattr(cl_sim, field)
            diff = np.max(np.abs(np_field - cl_field))

            # Non-zero validation
            self.assertGreater(np.max(np.abs(np_field)), 0.0, f"NumPy field {field} is zero!")
            # Precision tolerance
            self.assertLess(diff, 1e-4, f"Field mismatch in {field}: {diff:.6e}")

    def test_face_cpml_numpy_vs_opencl_engine_fields(self):
        """Face-local NumPy CPML must match OpenCL within the same field tolerance."""
        np_sim = NumPyFDTD_FaceCPML(self.shape, self.dl, npml=self.npml)
        cl_sim = OpenCLFDTD(self.shape, self.dl, npml=self.npml)
        np_sim.set_epsilon(self.eps)
        cl_sim.set_epsilon(self.eps)
        z_src = self.shape[2] - self.npml - 2

        def np_source(f):
            f.Ex[:, :, z_src] += np.sin(2 * np.pi * self.freq * f.t)

        def cl_source(f):
            f.add_source_Ex(z_src, np.sin(2 * np.pi * self.freq * f.t))

        np_sim.add_source(np_source)
        cl_sim.add_source(cl_source)
        for _ in range(50):
            np_sim.step()
            cl_sim.step()
        for field in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
            np_field = getattr(np_sim, field)
            cl_field = getattr(cl_sim, field)
            diff = np.max(np.abs(np_field - cl_field))
            self.assertGreater(np.max(np.abs(np_field)), 0.0, f"NumPy field {field} is zero!")
            self.assertLess(diff, 1e-4, f"Face-CPML mismatch in {field}: {diff:.6e}")

    def test_opencl_monitors(self):
        """GPU face-DFT + far-field should match host Near2FarBase on face interiors."""
        fdtd = OpenCLFDTD(self.shape, self.dl, npml=self.npml)
        fdtd.set_epsilon(self.eps)

        z_src = self.shape[2] - self.npml - 2
        fdtd.add_source(lambda f: f.add_source_Ex(z_src, np.sin(2 * np.pi * self.freq * f.t)))

        ctr_phys = (30e-3, 30e-3, 30e-3)
        size_phys = (20e-3, 20e-3, 20e-3)

        np_mon = NumPyNear2FarMonitor(fdtd, ctr_phys, size_phys, self.freq)
        cl_mon = OpenCLNear2FarMonitor(fdtd, ctr_phys, size_phys, self.freq)

        fdtd.run(80)

        obs_list = [
            (0.0, 0.0, 1000.0),
            (1000.0, 0.0, 0.0),
            (707.1, 0.0, 707.1),
        ]
        for obs in obs_list:
            ff_np = np_mon.get_farfield(obs)
            ff_cl = cl_mon.get_farfield(obs)
            diff = np.max(np.abs(ff_np - ff_cl))
            self.assertGreater(np.max(np.abs(ff_cl)), 0.0, f"OpenCL far-field at {obs} is zero")
            # Face-packed NumPy and OpenCL paths should now agree closely.
            self.assertLess(diff, 5e-4, f"Far-field mismatch at {obs}: {diff:.6e}")

        # Face-only download is much smaller than a full volume.
        n_cells = self.shape[0] * self.shape[1] * self.shape[2]
        self.assertLess(cl_mon.n_face_samples, n_cells // 2)

        # Polar helper returns finite dB samples.
        ang, db = cl_mon.farfield_polar_xz(distance_m=1000.0, n_angles=9)
        self.assertEqual(len(ang), 9)
        self.assertTrue(np.all(np.isfinite(db)))


if __name__ == "__main__":
    unittest.main()
