"""
Coverage for perfect-electric-conductor cells (materials.yee_edge_pec,
EngineFDTD.set_epsilon(pec_mask=)), added for foil-like reflectors whose
skin depth is far below the grid step (a finite sigma there would need the
skin depth resolved; see the set_epsilon docstring).
"""

from __future__ import annotations

import os
import unittest

import numpy as np
from opencl_fdtd_solver import OpenCLFDTD
from opencl_fdtd_solver.materials import yee_edge_pec


class TestYeeEdgePec(unittest.TestCase):
    def test_edge_or_logic(self):
        """An edge is PEC when either adjacent cell is PEC (closes the surface)."""
        m = np.zeros((3, 3, 3), dtype=bool)
        m[1, 1, 1] = True
        mx, my, mz = yee_edge_pec(m)
        # the masked cell itself
        self.assertTrue(mx[1, 1, 1] and my[1, 1, 1] and mz[1, 1, 1])
        # the edge one cell "before" along each stagger axis touches the PEC cell
        self.assertTrue(mx[0, 1, 1])
        self.assertTrue(my[1, 0, 1])
        self.assertTrue(mz[1, 1, 0])
        # edges not touching the cell stay active
        self.assertFalse(mx[2, 1, 1] or my[1, 2, 1] or mz[1, 1, 2])
        # 9 total per component: the 3x3 cell face behind + the cell itself... just
        # check the count is small and equal across components by symmetry
        self.assertEqual(mx.sum(), my.sum())
        self.assertEqual(my.sum(), mz.sum())

    def test_mask_input_not_mutated(self):
        m = np.zeros((4, 4, 4), dtype=bool)
        m[2, 2, 2] = True
        m0 = m.copy()
        yee_edge_pec(m)
        np.testing.assert_array_equal(m, m0)


class TestEnginePec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("PYOPENCL_CTX", "0")

    def test_pec_slab_blocks_and_reflects(self):
        """Plane-ish Ex wave onto a PEC slab: field behind the slab stays at
        numerical zero while the source side carries a standing wave."""
        shape = (24, 24, 40)
        dl = 2e-3
        npml = 4
        freq = 10e9
        fdtd = OpenCLFDTD(shape, dl, npml=npml)

        eps = np.ones(shape, dtype=np.float32)
        pec = np.zeros(shape, dtype=bool)
        z_slab = 16
        pec[:, :, z_slab : z_slab + 2] = True
        fdtd.set_epsilon(eps, pec_mask=pec)

        z_src = shape[2] - npml - 2
        fdtd.add_source(lambda f: f.add_source_Ex(z_src, np.sin(2 * np.pi * freq * f.t)))
        fdtd.run(160)

        c = shape[0] // 2
        inside = abs(fdtd.read_point("Ex", c, c, z_slab))
        behind = max(abs(fdtd.read_point("Ex", c, c, z)) for z in range(npml + 1, z_slab - 1))
        front = max(abs(fdtd.read_point("Ex", c, c, z)) for z in range(z_slab + 3, z_src - 1))
        self.assertEqual(inside, 0.0)
        self.assertGreater(front, 1e-3)
        self.assertLess(behind, front * 1e-6)

    def test_pec_with_sigma_array_coexists(self):
        """pec_mask must compose with the lossy path (both coefficient sets zeroed)."""
        shape = (12, 12, 12)
        fdtd = OpenCLFDTD(shape, 1e-3, npml=3)
        eps = np.ones(shape, dtype=np.float32)
        sigma = np.zeros(shape, dtype=np.float32)
        sigma[:, :, 4:6] = 5.0
        pec = np.zeros(shape, dtype=bool)
        pec[:, :, 8:9] = True
        fdtd.set_epsilon(eps, sigma_array=sigma, pec_mask=pec)
        fdtd.run(10)
        self.assertEqual(fdtd.read_point("Ex", 6, 6, 8), 0.0)

    def test_pec_mask_shape_mismatch_raises(self):
        """A mask that does not match the grid is rejected before any upload."""
        shape = (8, 8, 8)
        fdtd = OpenCLFDTD(shape, 1e-3, npml=2)
        eps = np.ones(shape, dtype=np.float32)
        with self.assertRaises(ValueError):
            fdtd.set_epsilon(eps, pec_mask=np.zeros((8, 8, 7), dtype=bool))


if __name__ == "__main__":
    unittest.main()
