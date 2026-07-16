# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
#
# This file is part of opencl-fdtd-solver.

"""Unit tests for CPML profile construction (κ grading, σ_max, E/H stagger)."""

from __future__ import annotations

import unittest

import numpy as np
from opencl_fdtd_solver.constants import ETA0
from opencl_fdtd_solver.cpml import (
    CPML_KAPPA_MAX,
    CPML_M,
    build_cpml_1d,
    build_cpml_profiles,
    sigma_max,
)


class TestCpmlProfiles(unittest.TestCase):
    def test_sigma_max_taflove(self):
        dl = 1e-3
        expect = 0.8 * (CPML_M + 1) / (ETA0 * dl)
        self.assertAlmostEqual(sigma_max(dl), expect, places=12)

    def test_kappa_grades_above_one(self):
        dl, dt, npml, n = 1e-3, 1e-12, 8, 40
        e = build_cpml_1d(n, npml=npml, dl=dl, dt=dt, node_offset=0.0)
        # Outer wall (index 0) should see κ → κ_max.
        self.assertGreater(float(e.kappa[0]), 1.0)
        self.assertAlmostEqual(float(e.kappa[0]), CPML_KAPPA_MAX, places=5)
        # Interior vacuum cells stay κ=1.
        self.assertAlmostEqual(float(e.kappa[n // 2]), 1.0, places=6)

    def test_eh_stagger_differs(self):
        dl, dt, npml, n = 1e-3, 1e-12, 8, 40
        e = build_cpml_1d(n, npml=npml, dl=dl, dt=dt, node_offset=0.0)
        h = build_cpml_1d(n, npml=npml, dl=dl, dt=dt, node_offset=0.5)
        # Same outer index, different depth → different σ/κ → different b.
        self.assertFalse(np.allclose(e.b[:npml], h.b[:npml]))
        self.assertFalse(np.allclose(e.kappa[:npml], h.kappa[:npml]))

    def test_build_profiles_shape(self):
        prof = build_cpml_profiles((16, 20, 24), npml=6, dl=2e-3, dt=1e-12)
        self.assertEqual(prof.e[0].b.shape, (16,))
        self.assertEqual(prof.h[2].kappa.shape, (24,))


if __name__ == "__main__":
    unittest.main()
