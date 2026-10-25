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
    def test_single_cell_clamps_its_twelve_boundary_edges(self):
        """One PEC cell masks exactly its 4 Ex, 4 Ey and 4 Ez cube edges.

        Yee stagger: Ex[i,j,k] at (i+½, j, k), Ey at (i, j+½, k), Ez at
        (i, j, k+½). Cell (1,1,1) spans [1,2]³, so its x-edges are the Ex
        samples at i=1 on the node lines (j,k) ∈ {1,2}², and so on.
        """
        m = np.zeros((4, 4, 4), dtype=bool)
        m[1, 1, 1] = True
        mx, my, mz = yee_edge_pec(m)

        ex = {(1, 1, 1), (1, 2, 1), (1, 1, 2), (1, 2, 2)}
        ey = {(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2)}
        ez = {(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1)}
        self.assertEqual(set(map(tuple, np.argwhere(mx))), ex)
        self.assertEqual(set(map(tuple, np.argwhere(my))), ey)
        self.assertEqual(set(map(tuple, np.argwhere(mz))), ez)
        # no expansion along a component's own axis: collinear edges outside
        # the cell stay active
        self.assertFalse(mx[0, 1, 1] or mx[2, 1, 1])
        self.assertFalse(my[1, 0, 1] or my[1, 2, 1])
        self.assertFalse(mz[1, 1, 0] or mz[1, 1, 2])

    def test_block_surface_is_closed(self):
        """Every boundary edge of a finite block is masked, interior edges too,
        and nothing outside the block's cube is touched."""
        m = np.zeros((8, 8, 8), dtype=bool)
        m[2:5, 2:5, 2:5] = True  # block spans [2,5]^3 in node coordinates
        mx, my, mz = yee_edge_pec(m)
        # Ex edges of the block: i in 2..4 (run across the cells), j,k in 2..5 (node lines)
        self.assertTrue(mx[2:5, 2:6, 2:6].all())
        self.assertEqual(mx.sum(), 3 * 4 * 4)
        self.assertTrue(my[2:6, 2:5, 2:6].all())
        self.assertEqual(my.sum(), 4 * 3 * 4)
        self.assertTrue(mz[2:6, 2:6, 2:5].all())
        self.assertEqual(mz.sum(), 4 * 4 * 3)

    def test_domain_boundary_is_clipped(self):
        """A PEC cell at index 0 has no predecessor; the mask must not wrap or fail."""
        m = np.zeros((3, 3, 3), dtype=bool)
        m[0, 0, 0] = True
        mx, my, mz = yee_edge_pec(m)
        self.assertEqual(
            set(map(tuple, np.argwhere(mx))), {(0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1)}
        )
        self.assertEqual(
            set(map(tuple, np.argwhere(my))), {(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)}
        )
        self.assertEqual(
            set(map(tuple, np.argwhere(mz))), {(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)}
        )

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

    def test_finite_block_has_no_active_tangential_edge(self):
        """Illuminate a finite PEC block; tangential E must be exactly zero on
        all six faces, including the +x/+y/+z faces that an axial-only mask
        expansion leaves free."""
        shape = (28, 28, 40)
        npml = 4
        fdtd = OpenCLFDTD(shape, 2e-3, npml=npml)
        eps = np.ones(shape, dtype=np.float32)
        pec = np.zeros(shape, dtype=bool)
        a, b = 10, 18  # cells a..b-1 → block occupies node range [a, b]
        c0, c1 = 14, 20
        pec[a:b, a:b, c0:c1] = True
        fdtd.set_epsilon(eps, pec_mask=pec)
        z_src = shape[2] - npml - 2
        fdtd.add_source(lambda f: f.add_source_Ex(z_src, np.sin(2 * np.pi * 10e9 * f.t)))
        fdtd.run(140)

        im, jm, km = (a + b) // 2, (a + b) // 2, (c0 + c1) // 2
        # Ex is tangential on the ±y and ±z faces
        for j in (a, b):
            self.assertEqual(fdtd.read_point("Ex", im, j, km), 0.0)
        for k in (c0, c1):
            self.assertEqual(fdtd.read_point("Ex", im, jm, k), 0.0)
        # Ey is tangential on the ±x and ±z faces
        for i in (a, b):
            self.assertEqual(fdtd.read_point("Ey", i, jm, km), 0.0)
        for k in (c0, c1):
            self.assertEqual(fdtd.read_point("Ey", im, jm, k), 0.0)
        # Ez is tangential on the ±x and ±y faces
        for i in (a, b):
            self.assertEqual(fdtd.read_point("Ez", i, jm, km), 0.0)
        for j in (a, b):
            self.assertEqual(fdtd.read_point("Ez", im, j, km), 0.0)
        # the wave did reach the block (the test is not vacuous)
        self.assertGreater(abs(fdtd.read_point("Ex", im, jm, c1 + 2)), 1e-4)

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
