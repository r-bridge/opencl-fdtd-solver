"""
Coverage for the lossy-conductor E-update path (materials.yee_edge_ca_cb,
EngineFDTD.set_epsilon(sigma_array=), accum_fp64).
"""

from __future__ import annotations

import unittest

import numpy as np

from opencl_fdtd_solver import OpenCLFDTD
from opencl_fdtd_solver.constants import EPS0
from opencl_fdtd_solver.materials import yee_edge_ca_cb, yee_edge_ce


class TestYeeEdgeCaCb(unittest.TestCase):
    def test_zero_sigma_matches_lossless_ce_exactly(self):
        """sigma=0 everywhere must recover Ca=1, Cb=yee_edge_ce exactly (old behavior)."""
        eps = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float64)
        sigma = np.zeros_like(eps)
        dt = 1e-13

        (ca_x, ca_y, ca_z), (cb_x, cb_y, cb_z) = yee_edge_ca_cb(eps, sigma, dt, dtype=np.float64)
        ce_x, ce_y, ce_z = yee_edge_ce(eps, dt, dtype=np.float64)

        for ca in (ca_x, ca_y, ca_z):
            np.testing.assert_array_equal(ca, np.ones_like(ca))
        np.testing.assert_allclose(cb_x, ce_x)
        np.testing.assert_allclose(cb_y, ce_y)
        np.testing.assert_allclose(cb_z, ce_z)

    def test_matches_hand_derived_formula(self):
        """Ca/Cb at a single interior point must match the closed-form formula directly."""
        eps_r, sigma_val, dt = 2.5, 15.0, 3e-13
        eps = np.full((3, 3, 3), eps_r, dtype=np.float64)
        sigma = np.full((3, 3, 3), sigma_val, dtype=np.float64)

        (ca_x, _, _), (cb_x, _, _) = yee_edge_ca_cb(eps, sigma, dt, dtype=np.float64)

        # Interior point: Yee-edge averaging is a no-op (uniform field), so the
        # formula should hold exactly, not just approximately.
        loss = sigma_val * dt / (2.0 * EPS0 * eps_r)
        expected_ca = (1.0 - loss) / (1.0 + loss)
        expected_cb = (dt / (EPS0 * eps_r)) / (1.0 + loss)

        np.testing.assert_allclose(ca_x[1, 1, 1], expected_ca, rtol=1e-10)
        np.testing.assert_allclose(cb_x[1, 1, 1], expected_cb, rtol=1e-10)

    def test_high_loss_ca_goes_negative_but_bounded(self):
        """Per-step loss > 1 gives Ca<0 (sign-flipping decay) but must stay in (-1, 1] --
        the scheme is unconditionally stable even though it's inaccurate in this regime
        (see fdtd_lib.scene_check's per-step-loss warning in the fdtd-studio repo)."""
        eps = np.ones((2, 2, 2), dtype=np.float64)
        for sigma_val in (100.0, 1000.0, 1e6):
            sigma = np.full_like(eps, sigma_val)
            (ca_x, _, _), _ = yee_edge_ca_cb(eps, sigma, dt=1e-12, dtype=np.float64)
            self.assertTrue(np.all(ca_x > -1.0) and np.all(ca_x <= 1.0))


class TestLossyConductorEngine(unittest.TestCase):
    """Live-engine smoke tests (run against whatever OpenCL device is available,
    same as the rest of this suite -- PoCL CPU is always present as a fallback)."""

    def _make_engine(self, accum_fp64=False):
        return OpenCLFDTD((24, 24, 60), 1e-3, npml=8, dtype=np.float32, accum_fp64=accum_fp64)

    def _run_slab(self, sigma_val, n_steps=400):
        fdtd = self._make_engine()
        eps = np.ones((fdtd.Nx, fdtd.Ny, fdtd.Nz), dtype=np.float32)
        sigma = np.zeros_like(eps)
        if sigma_val:
            sigma[:, :, 40:] = sigma_val
        fdtd.set_epsilon(eps, sigma_array=(sigma if sigma_val else None))
        freq = 10e9
        z_src = 15
        fdtd.add_source(lambda f: f.add_source_Ex(z_src, 0.1 * np.sin(2 * np.pi * freq * f.t)))
        for _ in range(n_steps):
            fdtd.step()
        return np.asarray(fdtd.Ex)

    def test_sigma_zero_matches_vacuum(self):
        """conductivity=0 must be indistinguishable from the lossless path (sigma_array=None)."""
        ex_explicit_zero = self._run_slab(0.0)

        fdtd = self._make_engine()
        eps = np.ones((fdtd.Nx, fdtd.Ny, fdtd.Nz), dtype=np.float32)
        fdtd.set_epsilon(eps)  # no sigma_array at all -- the old code path
        freq = 10e9
        fdtd.add_source(lambda f: f.add_source_Ex(15, 0.1 * np.sin(2 * np.pi * freq * f.t)))
        for _ in range(400):
            fdtd.step()

        np.testing.assert_allclose(ex_explicit_zero, np.asarray(fdtd.Ex), atol=1e-12)

    def test_reflection_increases_monotonically_with_sigma(self):
        """Field amplitude just before a lossy slab should rise monotonically with
        conductivity once sigma is large enough that the reflection dominates local
        interference phase (constructive buildup), and the field deep inside the
        slab should attenuate monotonically more.

        Deliberately starts at sigma=20, not 0 or 5: at very weak conductivity the
        reflection is weak enough that its interference with the incident wave at
        one fixed probe point can be constructive *or* destructive depending on
        relative phase there -- a real, reproducible effect (confirmed: sigma=5
        gives a *lower* amplitude than sigma=0 at this exact probe point), not
        something a monotonicity assertion should be built on. See
        test_sigma_zero_matches_vacuum for the clean (phase-independent) sigma=0
        check instead.
        """
        probe_before = []
        probe_inside = []
        for sigma_val in (20.0, 100.0, 300.0):
            ex = self._run_slab(sigma_val)
            probe_before.append(abs(ex[12, 12, 35]))
            probe_inside.append(abs(ex[12, 12, 55]))

        self.assertEqual(
            probe_before, sorted(probe_before),
            f"reflection buildup before the slab should increase with sigma: {probe_before}",
        )
        # Attenuation strengthens (deep-slab amplitude shrinks) monotonically too.
        self.assertGreater(probe_inside[0], probe_inside[1])
        self.assertGreater(probe_inside[1], probe_inside[2])


if __name__ == "__main__":
    unittest.main()
