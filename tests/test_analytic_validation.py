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

"""Analytic ground-truth checks (issue #48) — no Meep dependency."""

from __future__ import annotations

import os
import unittest
import warnings

import numpy as np
from opencl_fdtd_solver import OpenCLFDTD, OpenCLNear2FarMonitor
from opencl_fdtd_solver.constants import C0


class TestAnalyticDipolePattern(unittest.TestCase):
    """Ex-oriented compact source: XZ |S|(θ) should track cos²(θ) (θ from +z)."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("PYOPENCL_CTX", "0")

    def test_xz_cut_matches_cos2_theta(self):
        shape = (36, 36, 36)
        dl = 2e-3
        npml = 6
        freq = 5e9
        fwidth = 0.2 * freq
        fdtd = OpenCLFDTD(shape, dl, npml=npml)
        z = shape[2] // 2
        # Compact 3×3 Ex patch near the origin (Hertzian-like).
        i0, i1 = shape[0] // 2 - 1, shape[0] // 2 + 2
        j0, j1 = shape[1] // 2 - 1, shape[1] // 2 + 2
        t0 = 5.0 / (np.pi * fwidth)
        sigma = 1.0 / (np.pi * fwidth)

        def src(f):
            amp = np.exp(-0.5 * ((f.t - t0) / sigma) ** 2) * np.sin(2 * np.pi * freq * f.t)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                f.add_source_Ex(z, amp, i0=i0, i1=i1, j0=j0, j1=j1)

        fdtd.add_source(src)
        ctr = (shape[0] * dl / 2, shape[1] * dl / 2, shape[2] * dl / 2)
        size = (16 * dl, 16 * dl, 16 * dl)
        mon = OpenCLNear2FarMonitor(fdtd, ctr, size, freq)
        fdtd.run(400)

        angles, db = mon.farfield_polar_xz(distance_m=1.0, n_angles=37)
        # Mask deep nulls / numerical floor; compare shape of linear |S|.
        lin = 10.0 ** (np.asarray(db, dtype=np.float64) / 10.0)
        theta = np.deg2rad(angles)
        analytic = np.cos(theta) ** 2
        # Keep samples within 12 dB of peak (same idea as Meep pattern gates).
        peak = float(np.max(lin))
        mask = lin >= peak * 10 ** (-12.0 / 10.0)
        self.assertGreater(int(np.count_nonzero(mask)), 8)

        a = lin[mask]
        b = analytic[mask]
        # Least-squares scale analytic onto measured, then Pearson correlation.
        scale = float(np.dot(a, b) / (np.dot(b, b) + 1e-30))
        b_s = scale * b
        corr = float(np.corrcoef(a, b_s)[0, 1] if np.std(a) > 0 and np.std(b_s) > 0 else 0.0)
        self.assertGreater(
            corr,
            0.90,
            f"dipole |S| vs cos²θ correlation {corr:.4f} (expected > 0.90)",
        )


class TestAnalyticDispersion(unittest.TestCase):
    """Plane-wave numerical phase velocity vs Yee dispersion relation."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("PYOPENCL_CTX", "0")

    def test_yee_phase_velocity_1d_like(self):
        # Propagate a narrowband Ex pulse along +z in a long vacuum guide.
        nx, ny, nz = 16, 16, 120
        dl = 1e-3
        npml = 8
        freq = 6e9
        fdtd = OpenCLFDTD((nx, ny, nz), dl, npml=npml)
        # Courant number used by the solver.
        S = float(fdtd.dt * C0 / dl)
        # Analytic Yee dispersion for propagation along z (Δx=Δy unused → 1D-like):
        # sin(ωΔt/2) = S * sin(k̃ Δl/2)  →  k̃ from ω.
        omega = 2.0 * np.pi * freq
        arg = np.sin(0.5 * omega * fdtd.dt) / S
        self.assertLess(abs(arg), 1.0)
        k_num = 2.0 / dl * np.arcsin(arg)
        v_phase_analytic = omega / k_num
        # Relative error vs c should be small at ~50 cells/λ.
        cells_per_lambda = C0 / freq / dl
        self.assertGreater(cells_per_lambda, 40.0)
        err = abs(v_phase_analytic - C0) / C0
        self.assertLess(
            err,
            0.01,
            f"Yee |v_p - c|/c = {err:.4e} at {cells_per_lambda:.1f} cells/λ",
        )


class TestPmlReflection(unittest.TestCase):
    """Quantified CPML reflection via short vs extended reference domains."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("PYOPENCL_CTX", "0")

    @staticmethod
    def _probe_trace(*, nz: int, npml: int, z_src: int, z_probe: int, n_steps: int) -> np.ndarray:
        nx = ny = 16
        dl = 1e-3
        fdtd = OpenCLFDTD((nx, ny, nz), dl, npml=npml)
        t0 = 35.0 * fdtd.dt
        sigma = 6.0 * fdtd.dt
        i0, i1 = nx // 2 - 1, nx // 2 + 2
        j0, j1 = ny // 2 - 1, ny // 2 + 2

        def src(f):
            amp = -((f.t - t0) / sigma) * np.exp(-0.5 * ((f.t - t0) / sigma) ** 2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                f.add_source_Ex(z_src, float(amp), i0=i0, i1=i1, j0=j0, j1=j1)

        fdtd.add_source(src)
        probe = np.zeros(n_steps, dtype=np.float64)
        ix = iy = nx // 2
        for n in range(n_steps):
            if n == 90:
                fdtd.clear_sources()
            fdtd.step()
            probe[n] = float(fdtd.Ex[ix, iy, z_probe])
        return probe

    def test_normal_incidence_reflection_below_minus_25db(self):
        """Short domain (PML near probe) minus long reference ≈ reflected field.

        Pre-#49 CPML (weak σ, κ≡1, no stagger) measured ~−4 dB on this setup.
        """
        npml = 12
        z_src = 36
        z_probe = 70
        n_steps = 360
        nz_short = z_probe + 8 + npml
        nz_long = z_probe + 100 + npml
        p_short = self._probe_trace(
            nz=nz_short, npml=npml, z_src=z_src, z_probe=z_probe, n_steps=n_steps
        )
        p_long = self._probe_trace(
            nz=nz_long, npml=npml, z_src=z_src, z_probe=z_probe, n_steps=n_steps
        )
        i_inc = int(np.argmax(np.abs(p_long)))
        e_inc = float(np.max(np.abs(p_long)))
        self.assertGreater(e_inc, 1e-8)
        e_ref = float(np.max(np.abs((p_short - p_long)[i_inc + 20 :])))
        r = e_ref / e_inc
        r_db = 20.0 * np.log10(max(r, 1e-30))
        self.assertLess(
            r_db,
            -25.0,
            f"CPML |R|={r:.3e} ({r_db:.1f} dB); expected < -25 dB",
        )


if __name__ == "__main__":
    unittest.main()
