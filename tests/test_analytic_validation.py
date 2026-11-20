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
from opencl_fdtd_solver import NumPyNear2FarMonitor, OpenCLFDTD, OpenCLNear2FarMonitor
from opencl_fdtd_solver.constants import C0, EPS0, ETA0, MU0
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


class TestAnalyticHuygensEquivalence(unittest.TestCase):
    """Absolute SI Hertzian dipole, box invariance, and Poynting power conservation."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("PYOPENCL_CTX", "0")

    def test_hertzian_dipole_exact_si_amplitude_phase_power_and_box_invariance(self):
        """Verify closed-box Huygens quadrature against exact SI Hertzian dipole theory.

        Checks (with zero fitted scale or phase factors):
        1. Full 3-D complex vector E and H across 14 directions vs closed-form dipole.
        2. Broadside (+z) absolute amplitude (< 1%) and phase (< 1 deg) accuracy.
        3. Endfire null along ±x (< 1e-4).
        4. Surface-equivalence invariance: small centered cubic box vs larger shifted
           non-cubic box enclosing the same dipole.
        5. Poynting power conservation: near-field Huygens flux vs far-field sphere
           integral vs exact analytic dipole power (η₀/12π) k₀² |I·Δℓ|².
        6. OpenCLNear2FarMonitor vs NumPyNear2FarMonitor complex field parity.
        """
        shape = (40, 40, 40)
        dl = 2e-3
        npml = 8
        freq = 5e9
        omega = 2.0 * np.pi * freq
        k0 = omega / C0
        # Pulse centered at t0 = 5σ (~0.64 ns) decays to < 1e-12 by step 420 (~1.60 ns)
        # so the time-domain truncation error is negligible on both monitor boxes.
        fwidth = 0.4 * freq
        t0 = 4.0 / (np.pi * fwidth)
        sigma = 0.8 / (np.pi * fwidth)

        sim = OpenCLFDTD(shape, dl, npml=npml, dtype=np.float32)
        ic, jc, kc = 20, 20, 20
        j_dft = 0j

        def src(f):
            nonlocal j_dft
            amp = float(1e6 * np.exp(-0.5 * ((f.t - t0) / sigma) ** 2) * np.sin(omega * (f.t - t0)))
            f.add_source_Jx(kc, amp, i0=ic, i1=ic + 1, j0=jc, j1=jc + 1)
            j_dft += amp * np.exp(1j * omega * f.t) * f.dt

        sim.add_source(src)
        mon_a = OpenCLNear2FarMonitor(
            sim, (20 * dl, 20 * dl, 20 * dl), (8 * dl, 8 * dl, 8 * dl), freq
        )
        mon_b = OpenCLNear2FarMonitor(
            sim, (21 * dl, 19 * dl, 21 * dl), (12 * dl, 10 * dl, 14 * dl), freq
        )
        mon_np = NumPyNear2FarMonitor(
            sim, (20 * dl, 20 * dl, 20 * dl), (8 * dl, 8 * dl, 8 * dl), freq
        )
        sim.run(380)
        mon_a.snapshot_dft()
        mon_b.snapshot_dft()
        sim.run(70)
        # Verify both the source envelope and the monitor DFTs have fully converged.
        self.assertLess(float(np.exp(-0.5 * ((sim.t - t0) / sigma) ** 2)), 1e-10)
        self.assertLess(mon_a.dft_relative_change(), 1e-4)
        self.assertLess(mon_b.dft_relative_change(), 1e-4)

        dirs = np.array(
            [
                [0, 0, 1],
                [0, 0, -1],
                [0, 1, 0],
                [0, -1, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [1, 1, 1],
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
                [0.5, 0, 0.8660254],
                [0, 0.5, 0.8660254],
                [0.7071068, 0.7071068, 0],
                [0.3, -0.4, 0.8660254],
            ],
            dtype=np.float64,
        )
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        radius = 10.0
        obs = radius * dirs

        eh_a = mon_a.get_farfields(obs)
        eh_b = mon_b.get_farfields(obs)
        eh_np = np.array([mon_np.get_farfield(p) for p in obs])

        # Exact Hertzian dipole at Ex Yee edge (ic + 0.5, jc, kc) with dipole moment
        # I·dl = J_dft · dl³ in the exp(-iωt) convention:
        #   E_far(r) = (i k₀ η₀ / 4πr) exp(i k₀ (r - r̂·r_src)) (I·dl) (x̂ - (r̂·x̂)r̂).
        r_src = np.array([(ic + 0.5) * dl, jc * dl, kc * dl], dtype=np.float64)
        i_dl = j_dft * (dl**3)
        xhat = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        e_ana = np.zeros((len(dirs), 3), dtype=np.complex128)
        h_ana = np.zeros((len(dirs), 3), dtype=np.complex128)
        for m, rh in enumerate(dirs):
            xperp = xhat - np.dot(rh, xhat) * rh
            e_ana[m] = (
                (1j * k0 * ETA0 / (4.0 * np.pi * radius))
                * np.exp(1j * k0 * (radius - np.dot(rh, r_src)))
                * i_dl
                * xperp
            )
            h_ana[m] = np.cross(rh, e_ana[m]) / ETA0

        # 1. Exact 3-D complex vector E and H (no fitted scale or phase).
        rel_e = float(np.linalg.norm(eh_a[:, :3] - e_ana) / np.linalg.norm(e_ana))
        rel_h = float(np.linalg.norm(eh_a[:, 3:] - h_ana) / np.linalg.norm(h_ana))
        self.assertLess(rel_e, 1e-2, f"Hertzian dipole complex E rel L2 error {rel_e:.3e} >= 1%")
        self.assertLess(rel_h, 1e-2, f"Hertzian dipole complex H rel L2 error {rel_h:.3e} >= 1%")

        # 2. Broadside (+z) absolute amplitude and phase.
        mag_ratio = float(np.linalg.norm(eh_a[0, :3]) / np.linalg.norm(e_ana[0]))
        phase_err_deg = abs(float(np.rad2deg(np.angle(eh_a[0, 0] / e_ana[0, 0]))))
        self.assertLess(abs(mag_ratio - 1.0), 0.01, f"broadside |E| ratio {mag_ratio:.5f}")
        self.assertLess(phase_err_deg, 1.0, f"broadside phase error {phase_err_deg:.3f} deg")

        # 3. Endfire null along ±x.
        for idx in (4, 5):
            null_ratio = float(np.linalg.norm(eh_a[idx, :3]) / np.linalg.norm(eh_a[0, :3]))
            self.assertLess(null_ratio, 1e-4, f"endfire null ratio {null_ratio:.3e} >= 1e-4")

        # 4. Box-size and translation invariance (surface equivalence theorem).
        box_diff_e = float(np.linalg.norm(eh_a[:, :3] - eh_b[:, :3]) / np.linalg.norm(eh_a[:, :3]))
        box_diff_h = float(np.linalg.norm(eh_a[:, 3:] - eh_b[:, 3:]) / np.linalg.norm(eh_a[:, 3:]))
        self.assertLess(box_diff_e, 1e-2, f"Huygens box invariance E diff {box_diff_e:.3e} >= 1%")
        self.assertLess(box_diff_h, 1e-2, f"Huygens box invariance H diff {box_diff_h:.3e} >= 1%")

        # 5. Poynting power conservation: P_near == P_far == P_analytic.
        faces = mon_a.fetch_dft_fields()
        d_area = dl * dl
        p_near = 0.0
        for fid, (ax, sgn) in enumerate(((0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1))):
            sl = slice(mon_a._face_offsets[fid], mon_a._face_offsets[fid] + mon_a._face_counts[fid])
            ex_f, ey_f, ez_f = faces["Ex"][sl], faces["Ey"][sl], faces["Ez"][sl]
            hx_f, hy_f, hz_f = faces["Hx"][sl], faces["Hy"][sl], faces["Hz"][sl]
            if ax == 0:
                sn = 0.5 * np.real(ey_f * np.conj(hz_f) - ez_f * np.conj(hy_f))
            elif ax == 1:
                sn = 0.5 * np.real(ez_f * np.conj(hx_f) - ex_f * np.conj(hz_f))
            else:
                sn = 0.5 * np.real(ex_f * np.conj(hy_f) - ey_f * np.conj(hx_f))
            p_near += float(np.sum(sgn * sn) * d_area)

        mu_nodes, mu_w = np.polynomial.legendre.leggauss(20)
        n_phi = 40
        phi_nodes = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
        dphi = 2.0 * np.pi / n_phi
        pts_s = np.array(
            [
                [
                    radius * np.sqrt(max(0.0, 1.0 - mu * mu)) * np.cos(phi),
                    radius * np.sqrt(max(0.0, 1.0 - mu * mu)) * np.sin(phi),
                    radius * mu,
                ]
                for mu in mu_nodes
                for phi in phi_nodes
            ],
            dtype=np.float64,
        )
        w_s = np.repeat(mu_w, n_phi) * dphi
        eh_s = mon_a.get_farfields(pts_s)
        s_vec = 0.5 * np.real(np.cross(eh_s[:, :3], np.conj(eh_s[:, 3:])))
        s_r = np.sum(s_vec * (pts_s / radius), axis=1)
        p_far = float(np.sum(s_r * (radius**2) * w_s))
        p_ana = float((ETA0 / (12.0 * np.pi)) * (k0**2) * (abs(i_dl) ** 2))

        self.assertGreater(p_near, 0.0)
        self.assertLess(
            abs(p_far - p_near) / p_near,
            0.02,
            f"P_far ({p_far:.4e}) vs P_near ({p_near:.4e}) differs by > 2%",
        )
        self.assertLess(
            abs(p_far - p_ana) / p_ana,
            0.02,
            f"P_far ({p_far:.4e}) vs P_analytic ({p_ana:.4e}) differs by > 2%",
        )

        # 6. OpenCL vs NumPy monitor parity on full complex E, H. The floor is
        # relative to the peak (H in E units): components in a pattern null are
        # pure round-off when the device accumulates in FP32.
        scale = np.array([1, 1, 1, ETA0, ETA0, ETA0])
        peak = float(np.max(np.abs(eh_np * scale)))
        np.testing.assert_allclose(eh_a * scale, eh_np * scale, rtol=1e-4, atol=1e-6 * peak)


class TestAnalyticDispersion(unittest.TestCase):
    """Plane-wave numerical phase velocity vs Yee dispersion relation."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("PYOPENCL_CTX", "0")

    def test_yee_phase_velocity_1d_like(self):
        # Uniform XY sheets reduce the actual 3-D kernels to a 1-D wave.
        # Stop before reflections from either outer boundary reach the probes.
        freq = 6e9
        omega = 2.0 * np.pi * freq
        errors = []
        for dl in (2e-3, 1e-3):
            fdtd = OpenCLFDTD((1, 1, round(0.6 / dl)), dl, npml=0)
            z_src = round(0.2 / dl)
            probes = [round(0.23 / dl), round(0.24 / dl)]
            t0, width = 60 * fdtd.dt, 15 * fdtd.dt
            fdtd.add_source(
                lambda f: f.add_source_Jx(
                    z_src, np.exp(-0.5 * ((f.t - t0) / width) ** 2) * np.sin(omega * f.t)
                )
            )
            dft = np.zeros(2, dtype=complex)
            for _ in range(350):
                fdtd.step()
                dft += (
                    np.array([fdtd.read_point("Ex", 0, 0, z) for z in probes])
                    * np.exp(1j * omega * fdtd.t)
                    * fdtd.dt
                )
            self.assertGreater(np.min(np.abs(dft)), 1e-15)
            measured_k = np.angle(dft[1] / dft[0]) / ((probes[1] - probes[0]) * dl)
            courant = fdtd.dt * C0 / dl
            yee_k = 2 / dl * np.arcsin(np.sin(omega * fdtd.dt / 2) / courant)
            self.assertLess(abs(measured_k / yee_k - 1), 2e-4)
            errors.append(abs(omega / measured_k / C0 - 1))
        # Halving dl should reduce the second-order dispersion error by ~4.
        self.assertLess(errors[1], 0.35 * errors[0])


class TestPmlReflection(unittest.TestCase):
    """Quantified CPML reflection via short vs extended reference domains."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("PYOPENCL_CTX", "0")

    @staticmethod
    def _probe_trace(
        *,
        nz: int,
        npml: int,
        z_src: int,
        z_probe: int,
        n_steps: int,
        dl: float = 1e-3,
        interior_xy: int = 8,
    ) -> np.ndarray:
        nx = ny = max(16, 2 * npml + int(interior_xy))
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
            probe[n] = fdtd.read_point("Ex", ix, iy, z_probe)
        return probe

    @classmethod
    def _plus_z_reflection(
        cls,
        *,
        npml: int,
        dl: float = 1e-3,
        interior_xy: int = 8,
    ) -> tuple[float, np.ndarray]:
        # Offset source and probe by npml so BOTH the -z vacuum gap (24 cells)
        # and the +z vacuum gap (8 cells) stay fixed when sweeping npml.
        z_src = npml + 24
        z_probe = z_src + 34
        n_steps = 360
        p_short = cls._probe_trace(
            nz=z_probe + 8 + npml,
            npml=npml,
            z_src=z_src,
            z_probe=z_probe,
            n_steps=n_steps,
            dl=dl,
            interior_xy=interior_xy,
        )
        p_long = cls._probe_trace(
            nz=z_probe + 100 + npml,
            npml=npml,
            z_src=z_src,
            z_probe=z_probe,
            n_steps=n_steps,
            dl=dl,
            interior_xy=interior_xy,
        )
        i_inc = int(np.argmax(np.abs(p_long)))
        e_inc = float(np.max(np.abs(p_long)))
        if e_inc <= 1e-8:
            raise AssertionError(f"incident peak {e_inc:.3e} unexpectedly small")
        e_ref = float(np.max(np.abs((p_short - p_long)[i_inc + 20 :])))
        r_db = float(20.0 * np.log10(max(e_ref / e_inc, 1e-30)))
        return r_db, p_long

    def test_normal_incidence_reflection_below_minus_50db(self):
        """Short domain (PML near probe) minus long reference ≈ reflected field.

        Pre-#49 CPML (weak σ, κ≡1, no stagger) measured ~−4 dB on this setup.
        """
        r_db, _ = self._plus_z_reflection(npml=12)
        self.assertLess(
            r_db,
            -50.0,
            f"CPML |R|={r_db:.1f} dB; expected < -50 dB",
        )

    def test_low_and_high_z_pml_reflection_symmetric_and_below_minus_50db(self):
        """Both -z and +z CPML faces must absorb below -50 dB with matching performance."""
        npml = 12
        n_steps = 360
        r_p_db, _ = self._plus_z_reflection(npml=npml)
        # -z face (mirror geometry)
        z_probe_ms = npml + 8
        z_src_ms = z_probe_ms + 34
        p_s_m = self._probe_trace(
            nz=z_src_ms + 100 + npml, npml=npml, z_src=z_src_ms, z_probe=z_probe_ms, n_steps=n_steps
        )
        z_probe_ml = npml + 100
        z_src_ml = z_probe_ml + 34
        p_l_m = self._probe_trace(
            nz=z_src_ml + 100 + npml, npml=npml, z_src=z_src_ml, z_probe=z_probe_ml, n_steps=n_steps
        )
        i_inc_m = int(np.argmax(np.abs(p_l_m)))
        r_m_db = 20.0 * np.log10(
            max(
                float(np.max(np.abs((p_s_m - p_l_m)[i_inc_m + 20 :])))
                / float(np.max(np.abs(p_l_m))),
                1e-30,
            )
        )
        self.assertLess(r_p_db, -50.0, f"+z CPML reflection {r_p_db:.2f} dB; expected < -50 dB")
        self.assertLess(r_m_db, -50.0, f"-z CPML reflection {r_m_db:.2f} dB; expected < -50 dB")
        self.assertLess(
            abs(r_p_db - r_m_db),
            3.0,
            f"CPML -z ({r_m_db:.2f} dB) vs +z ({r_p_db:.2f} dB) asymmetry > 3 dB",
        )

    def test_pml_reflection_scale_invariant_across_dl(self):
        """End-to-end CPML |R| must be invariant across 6 orders of magnitude in dl."""
        r_opt_db, _ = self._plus_z_reflection(npml=12, dl=1e-6)
        r_vhf_db, _ = self._plus_z_reflection(npml=12, dl=1.0)
        self.assertLess(r_opt_db, -50.0, f"dl=1e-6 CPML |R|={r_opt_db:.2f} dB; expected < -50 dB")
        self.assertLess(r_vhf_db, -50.0, f"dl=1.0 CPML |R|={r_vhf_db:.2f} dB; expected < -50 dB")
        self.assertLess(
            abs(r_opt_db - r_vhf_db),
            1.0,
            f"CPML scale-invariance broken: dl=1e-6 ({r_opt_db:.2f} dB) vs dl=1.0 ({r_vhf_db:.2f} dB)",
        )

    def test_pml_reflection_converges_with_thickness(self):
        """Reflection drops monotonically with npml at fixed transverse vacuum interior."""
        r6, p_long6 = self._plus_z_reflection(npml=6, interior_xy=10)
        r8, p_long8 = self._plus_z_reflection(npml=8, interior_xy=10)
        r12, p_long12 = self._plus_z_reflection(npml=12, interior_xy=10)
        # Confirm the incident field arriving at the +z boundary is held fixed across npml.
        peak12 = float(np.max(np.abs(p_long12)))
        self.assertLess(float(np.max(np.abs(p_long6 - p_long12))) / peak12, 1e-2)
        self.assertLess(float(np.max(np.abs(p_long8 - p_long12))) / peak12, 1e-2)
        self.assertLess(
            r8, r6 - 5.0, f"expected npml=8 ({r8:.2f} dB) < npml=6 ({r6:.2f} dB) - 5 dB"
        )
        self.assertLess(
            r12, r8 - 10.0, f"expected npml=12 ({r12:.2f} dB) < npml=8 ({r8:.2f} dB) - 10 dB"
        )
        self.assertLess(r12, -65.0, f"npml=12 CPML |R|={r12:.2f} dB; expected < -65 dB")

    def test_oblique_45deg_single_face_pml_reflection(self):
        """Isolate pure 45° oblique reflection with path-matched PEC controls (front + back wall)."""
        dl = 1e-3
        npml = 10
        lam = 10.0 * dl
        freq = C0 / lam
        omega = 2.0 * np.pi * freq
        sin_t = float(np.sin(np.deg2rad(45.0)))

        # Steered 45° Gaussian beam in the y-z plane (Ex polarized, TE incidence on +z).
        # Three specular exit paths are calibrated against path-matched PEC mirrors at
        # z_pml_short (front), z_pml_short + npml (unstretched back wall), and
        # z_pml_short + 2*npml (κ-stretched back wall); p_long's +z boundary
        # (Δz = 76 + 74 = 150 cells -> ~262 steps) stays causally unreachable in 180 steps.
        nx = 2 * npml + 32
        ny = 2 * npml + 104
        z_src = npml + 16
        z_pml_short = z_src + 8
        z_probe = z_src + 2  # front specular Δz = 8 + 6 = 14 cells
        cx = nx // 2
        cy = npml + 20
        paths = (
            ("front", cy + 14, z_pml_short, slice(42, 80)),
            ("back_unstretched", cy + 34, z_pml_short + npml, slice(88, 132)),
            ("back_stretched", cy + 54, z_pml_short + 2 * npml, slice(136, 180)),
        )
        probe_ys = tuple(p[1] for p in paths)
        py_mir = cy - 14  # -45° mirror probe (verifies beam directionality)
        half_ap = 10
        w_cells = 6.0
        n_steps = 180

        def _run_beam(nz: int, *, pec_z: int | None = None) -> dict[int, np.ndarray]:
            fdtd = OpenCLFDTD((nx, ny, nz), dl, npml=npml)
            if pec_z is not None:
                eps = np.ones((nx, ny, nz), dtype=np.float32)
                pec = np.zeros((nx, ny, nz), dtype=bool)
                pec[:, :, pec_z] = True
                fdtd.set_epsilon(eps, pec_mask=pec)
            t0, sig = 25.0 * fdtd.dt, 5.0 * fdtd.dt
            js = np.arange(cy - half_ap, cy + half_ap + 1)
            weights = np.exp(-(((js - cy) / w_cells) ** 2))
            delays = (js - cy) * dl * sin_t / C0

            def src(f):
                for j, wt, tau in zip(js, weights, delays):
                    td = f.t - tau - t0
                    amp = float(wt * np.exp(-0.5 * (td / sig) ** 2) * np.sin(omega * td) * 1e5)
                    if abs(amp) > 1e-6:
                        f.add_source_Jx(z_src, amp, i0=cx - 4, i1=cx + 5, j0=int(j), j1=int(j) + 1)

            fdtd.add_source(src)
            traces = {py: np.zeros(n_steps, dtype=np.float64) for py in (*probe_ys, py_mir)}
            for step in range(n_steps):
                if step == 60:
                    fdtd.clear_sources()
                fdtd.step()
                for py in traces:
                    traces[py][step] = fdtd.read_point("Ex", cx, py, z_probe)
            return traces

        tr_short = _run_beam(z_pml_short + npml)
        tr_long = _run_beam(z_src + 76 + npml)

        win_all = slice(42, n_steps)
        for label, py, pz, win_pec in paths:
            tr_pec = _run_beam(pz + 2 + npml, pec_z=pz)
            e_ref_pec = float(np.max(np.abs((tr_pec[py] - tr_long[py])[win_pec])))
            self.assertGreater(e_ref_pec, 1e-6, f"missing PEC calibration on {label} path")
            if label == "front":
                e_ref_mir = float(np.max(np.abs((tr_pec[py_mir] - tr_long[py_mir])[win_pec])))
                # Confirm the phased aperture launches a genuinely directional +45° beam.
                self.assertGreater(
                    e_ref_pec,
                    20.0 * e_ref_mir,
                    f"steered beam not directional: +45°={e_ref_pec:.3e} vs -45°={e_ref_mir:.3e}",
                )
            e_ref_cpml = float(np.max(np.abs((tr_short[py] - tr_long[py])[win_all])))
            r_db = float(20.0 * np.log10(max(e_ref_cpml / e_ref_pec, 1e-30)))
            gate = -45.0 if label == "front" else -38.0
            self.assertLess(
                r_db,
                gate,
                f"45° steered-beam +z CPML ({label}) reflection {r_db:.2f} dB; expected < {gate} dB",
            )

    def test_3d_enclosure_reflection_and_symmetry(self):
        """Total reflected field of a 3-D CPML enclosure (faces, edges, corners) is low & symmetric."""
        dl = 1e-3
        npml = 8
        n_short = 28  # interior [8..20), outer-wall corner round-trip ~40 cells (~70 steps)
        pad = 24  # n_long = 76 -> wall round-trip >= 70 cells > 120 * 0.5716 = 68.6 cells
        n_steps = 120  # t0=25 + 70 = 95 -> captures full outermost-corner return & tail
        c_s = n_short // 2
        c_l = c_s + pad
        offsets = {
            "+z": (0, 0, 4),
            "-z": (0, 0, -4),
            "+y": (0, 4, 0),
            "-y": (0, -4, 0),
            "+x+z": (4, 0, 4),
            "-x+z": (-4, 0, 4),
            "+y+z": (0, 4, 4),
            "-y-z": (0, -4, -4),
            "+x+y+z": (4, 4, 4),
            "-x-y-z": (-4, -4, -4),
        }

        def _run_3d(n: int, p: int, c: int) -> dict[str, np.ndarray]:
            sim = OpenCLFDTD((n, n, n), dl, npml=p)
            t0 = 25.0 * sim.dt
            sig = 5.0 * sim.dt

            def src(f):
                amp = -((f.t - t0) / sig) * np.exp(-0.5 * ((f.t - t0) / sig) ** 2)
                f.add_source_Jx(c, float(amp) * 1e5, i0=c, i1=c + 1, j0=c, j1=c + 1)

            sim.add_source(src)
            traces = {k: np.zeros(n_steps, dtype=np.float64) for k in offsets}
            for step in range(n_steps):
                if step == 60:
                    sim.clear_sources()
                sim.step()
                for k, (di, dj, dk) in offsets.items():
                    traces[k][step] = sim.read_point("Ex", c + di, c + dj, c + dk)
            return traces

        tr_s = _run_3d(n_short, npml, c_s)
        tr_l = _run_3d(n_short + 2 * pad, 0, c_l)
        refl_db = {}
        for k in offsets:
            inc = float(np.max(np.abs(tr_l[k])))
            self.assertGreater(inc, 1e-6, f"probe {k} saw negligible incident wave")
            ref = float(np.max(np.abs(tr_s[k] - tr_l[k])))
            refl_db[k] = float(20.0 * np.log10(max(ref / inc, 1e-30)))

        # Bound total enclosure reflection at face-adjacent, edge-adjacent, and corner-adjacent probes.
        for k in ("+z", "-z", "+y", "-y"):
            self.assertLess(
                refl_db[k],
                -45.0,
                f"face-adjacent {k} total reflection {refl_db[k]:.2f} dB >= -45 dB",
            )
        for k in ("+x+z", "-x+z", "+y+z", "-y-z"):
            self.assertLess(
                refl_db[k],
                -40.0,
                f"edge-adjacent {k} total reflection {refl_db[k]:.2f} dB >= -40 dB",
            )
        for k in ("+x+y+z", "-x-y-z"):
            self.assertLess(
                refl_db[k],
                -35.0,
                f"corner-adjacent {k} total reflection {refl_db[k]:.2f} dB >= -35 dB",
            )
        # Exact axis/reflection symmetries (y <-> z and ±x) match tightly; opposing
        # ±y/±z probes include a 0.5-cell Yee offset relative to the PML interface (~2 dB).
        self.assertLess(abs(refl_db["+z"] - refl_db["+y"]), 0.5)
        self.assertLess(abs(refl_db["-z"] - refl_db["-y"]), 0.5)
        self.assertLess(abs(refl_db["+x+z"] - refl_db["-x+z"]), 0.5)
        self.assertLess(abs(refl_db["+y+z"] - refl_db["-y-z"]), 3.0)
        self.assertLess(abs(refl_db["+x+y+z"] - refl_db["-x-y-z"]), 3.0)


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

    def _run_case(self, *, with_sphere: bool):
        shape = (48, 48, 56)
        dl = 2e-3
        npml = 8
        freq = 5e9
        fwidth = 0.25 * freq
        eps_r = 4.0
        rad_cells = 6

        fdtd = OpenCLFDTD(shape, dl, npml=npml)
        eps = np.ones(shape, dtype=np.float32)
        if with_sphere:
            ctr = np.array(shape) // 2
            ii, jj, kk = np.ogrid[: shape[0], : shape[1], : shape[2]]
            inside = (ii - ctr[0]) ** 2 + (jj - ctr[1]) ** 2 + (kk - ctr[2]) ** 2 < rad_cells**2
            eps[inside] = eps_r
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
        ctr_phys = tuple(0.5 * c * dl for c in shape)
        half = (rad_cells + 4) * dl
        size = (2 * half, 2 * half, 2 * half)
        mon = OpenCLNear2FarMonitor(fdtd, ctr_phys, size, freq)
        fdtd.run(500)

        angles_deg = np.linspace(-180.0, 180.0, 37)
        rad = np.deg2rad(angles_deg)
        pts = np.column_stack([np.sin(rad), np.zeros_like(rad), np.cos(rad)])  # R = 1 m
        return angles_deg, mon.get_farfields(pts)

    def test_eplane_pattern_matches_mie(self):
        # Moderate-size dielectric sphere: ka ≈ 1.26, ~6 cells / radius.
        dl = 2e-3
        freq = 5e9
        eps_r = 4.0
        rad_cells = 6
        a = rad_cells * dl
        k0 = 2.0 * np.pi * freq / C0
        ka = k0 * a
        self.assertGreater(ka, 1.0)
        self.assertLess(ka, 2.0)

        # True scattered field via empty subtraction (complex EH, sphere run
        # minus identical vacuum run): the finite rim-tapered source sheet's
        # own through-going wave does not cancel perfectly on the Huygens box,
        # and its leakage lands in the forward direction — right on top of
        # the Mie forward lobe — without this subtraction.
        angles_deg, eh_sphere = self._run_case(with_sphere=True)
        _, eh_empty = self._run_case(with_sphere=False)
        scat = eh_sphere - eh_empty
        E, H = scat[:, 0:3], scat[:, 3:6]
        Sx = 0.5 * (E[:, 1] * np.conj(H[:, 2]) - E[:, 2] * np.conj(H[:, 1]))
        Sy = 0.5 * (E[:, 2] * np.conj(H[:, 0]) - E[:, 0] * np.conj(H[:, 2]))
        Sz = 0.5 * (E[:, 0] * np.conj(H[:, 1]) - E[:, 1] * np.conj(H[:, 0]))
        lin = np.sqrt(np.abs(Sx) ** 2 + np.abs(Sy) ** 2 + np.abs(Sz) ** 2)

        # Illumination propagates +z, so Mie forward scattering (the dominant
        # lobe at this ka) radiates toward +z = θ=0: scattering angle |θ|.
        # NOTE: an earlier revision of this test mapped θ_mie = π−|θ| ("FDTD
        # peak near ±180° ↔ Mie forward") — that accommodated an inverted
        # one-sidedness in the near-to-far kernel's N/L combination (the
        # forward lobe was landing at the backward angle) rather than
        # questioning it. The kernel inversion is now fixed (see
        # dft_farfield.cl, farfield_nl_to_eh) and this test uses the physical
        # mapping. Absolute RCS is float32-limited and is not gated here.
        theta_mie = np.deg2rad(np.abs(angles_deg))
        mie = mie_eplane_intensity(np.sqrt(eps_r), ka, theta_mie)
        peak = float(np.max(lin))
        self.assertGreater(peak, 0.0)
        peak_angle = float(angles_deg[int(np.argmax(lin))])
        self.assertLess(
            abs(peak_angle),
            15.0,
            f"forward-scattering peak should be at θ≈0 (+z, the propagation "
            f"direction), got {peak_angle:.1f}°",
        )
        mask = lin >= peak * 10 ** (-12.0 / 10.0)
        self.assertGreater(int(np.count_nonzero(mask)), 8)

        a_m = lin[mask]
        b_m = mie[mask]
        scale = float(np.dot(a_m, b_m) / (np.dot(b_m, b_m) + 1e-30))
        b_s = scale * b_m
        corr = float(np.corrcoef(a_m, b_s)[0, 1] if np.std(a_m) > 0 and np.std(b_s) > 0 else 0.0)
        self.assertGreater(
            corr,
            0.95,
            f"Mie E-plane |S| correlation {corr:.4f} (ka={ka:.3f}, expected > 0.95)",
        )


if __name__ == "__main__":
    unittest.main()
