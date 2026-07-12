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
import pyopencl as cl
from opencl_fdtd_solver import OpenCLFDTD, NumPyFDTD, NumPyNear2FarMonitor, OpenCLNear2FarMonitor


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
                    if (i - ctr)**2 + (j - ctr)**2 + (k - ctr)**2 < rad**2:
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

        np_sim._sources.append(np_source)
        cl_sim._sources.append(cl_source)

        # Run both for 50 steps
        for _ in range(50):
            np_sim.step()
            cl_sim.step()

        # Check maximum difference across all field components
        for field in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']:
            np_field = getattr(np_sim, field)
            cl_field = getattr(cl_sim, field)
            diff = np.max(np.abs(np_field - cl_field))
            
            # Non-zero validation
            self.assertGreater(np.max(np.abs(np_field)), 0.0, f"NumPy field {field} is zero!")
            # Precision tolerance
            self.assertLess(diff, 1e-4, f"Field mismatch in {field}: {diff:.6e}")

    def test_opencl_monitors(self):
        """Compare GPU DFT accumulation (OpenCLNear2FarMonitor) with host CPU accumulation."""
        fdtd = OpenCLFDTD(self.shape, self.dl, npml=self.npml)
        fdtd.set_epsilon(self.eps)

        z_src = self.shape[2] - self.npml - 2
        fdtd._sources.append(lambda f: f.add_source_Ex(z_src, np.sin(2 * np.pi * self.freq * f.t)))

        # Define Near2Far Huygens box coordinates
        ctr_phys = (30e-3, 30e-3, 30e-3)
        size_phys = (20e-3, 20e-3, 20e-3)

        np_mon = NumPyNear2FarMonitor(fdtd, ctr_phys, size_phys, self.freq)
        cl_mon = OpenCLNear2FarMonitor(fdtd, ctr_phys, size_phys, self.freq)

        fdtd.run(80)
        cl_mon.fetch_dft_fields()

        # Ensure GPU DFT fields match host DFT fields exactly
        for name in ['Ex_dft', 'Ey_dft', 'Ez_dft', 'Hx_dft', 'Hy_dft', 'Hz_dft']:
            np_arr = getattr(np_mon, name)
            cl_arr = getattr(cl_mon, name)
            diff = np.max(np.abs(np_arr - cl_arr))
            
            self.assertGreater(np.max(np.abs(np_arr)), 0.0, f"Monitor DFT {name} is zero!")
            self.assertLess(diff, 1e-4, f"Monitor DFT mismatch in {name}: {diff:.6e}")


if __name__ == '__main__':
    unittest.main()
