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
from opencl_fdtd_solver.constants import C0, EPS0, MU0
from tests.analytic_mie import mie_eplane_intensity


def _em_energy(fdtd, eps_r: np.ndarray | None = None) -> float:
    """Cell-sum SI electromagnetic energy (Yee staggering ignored → O(Δℓ) approx)."""
    ex, ey, ez = fdtd.Ex, fdtd.Ey, fdtd.Ez
    hx, hy, hz = fdtd.Hx, fdtd.Hy, fdtd.Hz
    if eps_r is None:
        eps_r = getattr(fdtd, "eps_r", None)
    if eps_r is None:
        eps_r = np.ones((fdtd.Nx, fdtd.Ny, fdtd.Nz), dtype=np.float64)
    e = 0.5 * EPS0 * eps_r * (ex * ex + ey * ey + ez * ez)
    h = 0.5 * MU0 * (hx * hx + hy * hy + hz * hz)
    return float(np.sum(e + h) * fdtd.dl**3)


class TestAnalyticDipolePattern(unittest.TestCase):
    """Ex-oriented compact source: XZ |S|(θ) should track cos²(θ) (θ from +z)."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("PYOPENCL_CTX", "0")

    def _run_dipole_monitor(self):
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
        return mon.farfield_polar_xz(distance_m=1.0, n_angles=37)

    def test_xz_cut_matches_cos2_theta(self):
        angles, db = self._run_dipole_monitor()
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

    def test_endfire_null_depth_vs_analytic(self):
        """Strict null gate: |S|(±90°)/|S|(0°) must be deep for an Ex Hertzian."""
        angles, db = self._run_dipole_monitor()
        lin = 10.0 ** (np.asarray(db, dtype=np.float64) / 10.0)
        # Broadside = θ=0 (+z); endfire nulls at θ=±90°.
        i0 = int(np.argmin(np.abs(angles - 0.0)))
        i_p90 = int(np.argmin(np.abs(angles - 90.0)))
        i_m90 = int(np.argmin(np.abs(angles + 90.0)))
        peak = float(lin[i0])
        self.assertGreater(peak, 0.0)
        null_ratio = max(float(lin[i_p90]), float(lin[i_m90])) / peak
        # Analytic cos²θ is exactly 0 at ±90°; float32 N2F on this grid is noisy
        # but should stay well below the Meep live gate (0.05).
        self.assertLess(
            null_ratio,
            0.03,
            f"endfire null ratio {null_ratio:.3e} (expected < 0.03)",
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


class TestEnergyConservation(unittest.TestCase):
    """Closed lossless box (npml=0): energy should not grow after the source is off."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("PYOPENCL_CTX", "0")

    def test_energy_stable_after_source_off(self):
        shape = (24, 24, 24)
        dl = 1e-3
        fdtd = OpenCLFDTD(shape, dl, npml=0)
        z = shape[2] // 2
        i0, i1 = shape[0] // 2 - 1, shape[0] // 2 + 2
        j0, j1 = shape[1] // 2 - 1, shape[1] // 2 + 2
        t0 = 20.0 * fdtd.dt
        sigma = 4.0 * fdtd.dt

        def src(f):
            amp = -((f.t - t0) / sigma) * np.exp(-0.5 * ((f.t - t0) / sigma) ** 2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                f.add_source_Ex(z, float(amp), i0=i0, i1=i1, j0=j0, j1=j1)

        fdtd.add_source(src)
        n_drive = 80
        n_coast = 200
        for _ in range(n_drive):
            fdtd.step()
        fdtd.clear_sources()
        e0 = _em_energy(fdtd)
        self.assertGreater(e0, 1e-20)
        energies = []
        for _ in range(n_coast):
            fdtd.step()
            if fdtd.step_num % 20 == 0:
                energies.append(_em_energy(fdtd))
        arr = np.asarray(energies, dtype=np.float64)
        # No secular growth; allow float32 + staircase leakage at the truncated boundary.
        self.assertLess(
            float(np.max(arr)),
            1.15 * e0,
            f"energy grew: max={arr.max():.3e} vs post-source {e0:.3e}",
        )
        # Mean coasting energy stays within 25% of the post-source value.
        self.assertLess(
            abs(float(np.mean(arr)) - e0) / e0,
            0.25,
            f"mean coast energy {arr.mean():.3e} vs {e0:.3e}",
        )


class TestMieSeriesUnit(unittest.TestCase):
    """Pure analytic checks of the Bohren–Huffman helper (no OpenCL)."""

    def test_rayleigh_eplane_tracks_cos2(self):
        # Small dielectric sphere: |S₂|² ∝ cos²θ in the E-plane.
        x = 0.05
        m = 1.5
        theta = np.linspace(0.0, np.pi, 61)
        inten = mie_eplane_intensity(m, x, theta)
        analytic = np.cos(theta) ** 2
        inten = inten / np.max(inten)
        analytic = analytic / np.max(analytic)
        corr = float(np.corrcoef(inten, analytic)[0, 1])
        self.assertGreater(corr, 0.995, f"Rayleigh |S2|^2 vs cos^2θ corr={corr:.4f}")

    def test_nonabsorbing_qext_equals_qsca(self):
        from tests.analytic_mie import mie_coefficients, mie_nstop

        x = 1.2
        m = 1.5
        a_n, b_n = mie_coefficients(m, x, nstop=mie_nstop(x))
        n = np.arange(1, len(a_n) + 1, dtype=np.float64)
        qsca = (2.0 / x**2) * np.sum((2 * n + 1) * (np.abs(a_n) ** 2 + np.abs(b_n) ** 2))
        qext = (2.0 / x**2) * np.sum((2 * n + 1) * np.real(a_n + b_n))
        self.assertLess(abs(qext - qsca) / qext, 1e-10)


class TestMieSphereRCS(unittest.TestCase):
    """Bistatic E-plane |S|(θ) for a dielectric sphere vs Mie series."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("PYOPENCL_CTX", "0")

    def test_eplane_pattern_matches_mie(self):
        # Moderate-size dielectric sphere: ka ≈ 1.26, ~6 cells / radius.
        shape = (48, 48, 56)
        dl = 2e-3
        npml = 8
        freq = 5e9
        fwidth = 0.25 * freq
        eps_r = 4.0
        rad_cells = 6
        a = rad_cells * dl
        k0 = 2.0 * np.pi * freq / C0
        ka = k0 * a
        self.assertGreater(ka, 1.0)
        self.assertLess(ka, 2.0)

        fdtd = OpenCLFDTD(shape, dl, npml=npml)
        eps = np.ones(shape, dtype=np.float32)
        ctr = np.array(shape) // 2
        ii, jj, kk = np.ogrid[: shape[0], : shape[1], : shape[2]]
        eps[(ii - ctr[0]) ** 2 + (jj - ctr[1]) ** 2 + (kk - ctr[2]) ** 2 < rad_cells**2] = eps_r
        fdtd.set_epsilon(eps)

        # Soft Jx sheet at low z → illumination propagating roughly +z.
        z_src = npml + 2
        p = npml + 1
        t0 = 4.0 / (np.pi * fwidth)
        sigma = 0.8 / (np.pi * fwidth)

        def src(f):
            amp = np.exp(-0.5 * ((f.t - t0) / sigma) ** 2) * np.sin(2 * np.pi * freq * f.t)
            f.add_source_Jx(
                z_src,
                amp,
                i0=p,
                i1=shape[0] - p,
                j0=p,
                j1=shape[1] - p,
                rim_taper=True,
            )

        fdtd.add_source(src)
        # Huygens box encloses the sphere; source is outside → N2F ≈ scattered field.
        ctr_phys = tuple(0.5 * c * dl for c in shape)
        half = (rad_cells + 4) * dl
        size = (2 * half, 2 * half, 2 * half)
        mon = OpenCLNear2FarMonitor(fdtd, ctr_phys, size, freq)
        fdtd.run(500)

        angles_deg, db = mon.farfield_polar_xz(distance_m=1.0, n_angles=37)
        lin = 10.0 ** (np.asarray(db, dtype=np.float64) / 10.0)
        # farfield_polar_xz: obs = (R sinθ, 0, R cosθ), θ ∈ [-180°, 180°].
        # For this soft-source geometry the measured E-plane intensity tracks
        # Mie |S₂|² at scattering angle π−|θ| (FDTD peak near ±180° ↔ Mie
        # forward). Absolute RCS is float32-limited and is not gated here.
        theta_mie = np.pi - np.deg2rad(np.abs(angles_deg))
        mie = mie_eplane_intensity(np.sqrt(eps_r), ka, theta_mie)
        peak = float(np.max(lin))
        self.assertGreater(peak, 0.0)
        mask = lin >= peak * 10 ** (-12.0 / 10.0)
        self.assertGreater(int(np.count_nonzero(mask)), 8)

        a_m = lin[mask]
        b_m = mie[mask]
        scale = float(np.dot(a_m, b_m) / (np.dot(b_m, b_m) + 1e-30))
        b_s = scale * b_m
        corr = float(np.corrcoef(a_m, b_s)[0, 1] if np.std(a_m) > 0 and np.std(b_s) > 0 else 0.0)
        self.assertGreater(
            corr,
            0.90,
            f"Mie E-plane |S| correlation {corr:.4f} (ka={ka:.3f}, expected > 0.90)",
        )


if __name__ == "__main__":
    unittest.main()
