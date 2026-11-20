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
        self.assertFalse(np.allclose(e.b[n - npml :], h.b[n - npml :]))
        self.assertFalse(np.allclose(e.kappa[n - npml :], h.kappa[n - npml :]))

    def test_high_side_is_dual_grid_mirror_of_low_side(self):
        """High-side H[n-npml+i] sits deeper than E[n-npml+i], mirroring low-side E/H."""
        dl, dt, npml, n = 1e-3, 1e-12, 8, 40
        e = build_cpml_1d(n, npml=npml, dl=dl, dt=dt, node_offset=0.0, dtype=np.float64)
        h = build_cpml_1d(n, npml=npml, dl=dl, dt=dt, node_offset=0.5, dtype=np.float64)
        # Low-side E[npml-1-i] has depth (i+1)/npml == high-side H[n-npml+i].
        np.testing.assert_allclose(e.kappa[:npml][::-1], h.kappa[n - npml :], rtol=1e-12)
        np.testing.assert_allclose(e.b[:npml][::-1], h.b[n - npml :], rtol=1e-12)
        # Low-side H[npml-1-i] has depth (i+0.5)/npml == high-side E[n-npml+i].
        np.testing.assert_allclose(h.kappa[:npml][::-1], e.kappa[n - npml :], rtol=1e-12)
        np.testing.assert_allclose(h.b[:npml][::-1], e.b[n - npml :], rtol=1e-12)

    def test_alpha_max_scales_inversely_with_dl(self):
        """CFS decay b at the vacuum interface must be scale-invariant when dt ~ dl."""
        npml, n = 8, 40
        p1 = build_cpml_1d(n, npml=npml, dl=1e-3, dt=1e-12, node_offset=0.5, dtype=np.float64)
        p2 = build_cpml_1d(n, npml=npml, dl=1e-6, dt=1e-15, node_offset=0.5, dtype=np.float64)
        np.testing.assert_allclose(p1.b, p2.b, rtol=1e-12)
        np.testing.assert_allclose(p1.kappa, p2.kappa, rtol=1e-12)
        np.testing.assert_allclose(p1.c * 1e-3, p2.c * 1e-6, rtol=1e-12)

    def test_build_profiles_shape(self):
        prof = build_cpml_profiles((16, 20, 24), npml=6, dl=2e-3, dt=1e-12)
        self.assertEqual(prof.e[0].b.shape, (16,))
        self.assertEqual(prof.h[2].kappa.shape, (24,))


if __name__ == "__main__":
    unittest.main()
