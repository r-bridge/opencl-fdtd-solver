# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
#
# This file is part of opencl-fdtd-solver.

"""
Ground-truth regression test for a confirmed near-to-far reconstruction bug.

No FDTD physics is involved: the six Huygens-box face DFT buffers of an
``OpenCLNear2FarMonitor`` are hand-populated with the exact analytic field of
a known plane wave crossing the box at a chosen angle (Ey-polarized,
``E = y_hat * exp(-jk(sin(theta)*x + cos(theta)*z))``, ``H = k_hat x E /
eta0``, using the same corner-origin absolute grid coordinates the kernel
itself uses for its face-sample positions).

By the equivalence principle, a closed box enclosing a source-free plane
wave must reconstruct a far-field peak at exactly that wave's own
propagation angle -- this is a basic, well-established property, independent
of any FDTD numerics, grid resolution, or material model. It is the cleanest
possible ground truth for validating (or, currently, falsifying) the
far-field integral in isolation from all FDTD/geometry confounds.

**Current status: this test is expected to fail.** The reconstruction
consistently produces a *null* exactly at the true propagation direction and
a spurious peak elsewhere, for every non-trivial (non-broadside-symmetric)
angle tested. A candidate fix (flipping the sign of the r_hat x L term in
``farfield_nl_to_eh``, matching the standard Balanis eq. 12-30a/b combined
formula) was tried and, while it corrects the single-face broadside
degenerate case, was found to make the *closed*-box reconstruction and the
existing Meep cross-validation baselines (``test_meep_farfield_baselines.py``
-- see ``opencl_null_ratio`` vs ``meep_null_ratio`` in
``tests/meep_validation/baselines/DISCREPANCY_REPORT_FARFIELD.md``, both
before and after that candidate fix, off by ~1000x) no better or slightly
worse. The root cause is therefore NOT the N/L combination sign alone --
left as `expectedFailure` with this note rather than merged as an unverified
fix. Whoever picks this up next: use ``fill_plane_wave`` below as the
ground-truth harness to iterate against; do not trust a fix that isn't
verified to also improve (or at least not worsen) the existing Meep
baselines' null-ratio metrics.
"""

from __future__ import annotations

import unittest

import numpy as np
import pyopencl as cl

from opencl_fdtd_solver import OpenCLFDTD, OpenCLNear2FarMonitor
from opencl_fdtd_solver.constants import C0, ETA0

from tests.meep_validation.harness import ensure_pyopencl_ctx


def _fill_analytic_plane_wave(mon, dl: float, k: float, theta_deg: float) -> None:
    """Populate all 6 face DFT buffers with an Ey-polarized plane wave
    travelling at ``theta_deg`` from +Z in the XZ plane, using the exact
    corner-origin absolute grid coordinates ``face_sample_NL`` itself uses.
    """
    th = np.deg2rad(theta_deg)
    sx, cz = np.sin(th), np.cos(th)

    def phase_at(i, j, k_idx):
        x = i * dl
        z = k_idx * dl
        return np.exp(-1j * k * (sx * x + cz * z))

    acc_dtype = mon._acc_dtype
    nyf, nzf = mon.nyf, mon.nzf
    offs = mon._face_offsets
    n_total = mon.n_face_samples

    Ex = np.zeros(n_total, dtype=np.complex128)
    Ey = np.zeros(n_total, dtype=np.complex128)
    Ez = np.zeros(n_total, dtype=np.complex128)
    Hx = np.zeros(n_total, dtype=np.complex128)
    Hy = np.zeros(n_total, dtype=np.complex128)
    Hz = np.zeros(n_total, dtype=np.complex128)

    def set_face(face_id, off, count):
        for loc in range(count):
            if face_id in (0, 1):
                j_l, k_l = loc // nzf, loc % nzf
                abs_i = mon.ix0 if face_id == 0 else mon.ix1
                abs_j = mon.iy0 + j_l
                abs_k = mon.iz0 + k_l
            elif face_id in (2, 3):
                i_l, k_l = loc // nzf, loc % nzf
                abs_i = mon.ix0 + i_l
                abs_j = mon.iy0 if face_id == 2 else mon.iy1
                abs_k = mon.iz0 + k_l
            else:
                i_l, j_l = loc // nyf, loc % nyf
                abs_i = mon.ix0 + i_l
                abs_j = mon.iy0 + j_l
                abs_k = mon.iz0 if face_id == 4 else mon.iz1
            ph = phase_at(abs_i, abs_j, abs_k)
            idx = off + loc
            Ey[idx] = ph
            Hx[idx] = -cz * ph / ETA0
            Hz[idx] = sx * ph / ETA0

    for fid in range(6):
        set_face(fid, offs[fid], mon._face_counts[fid])

    def upload(buf, arr):
        host = np.empty(n_total * 2, dtype=acc_dtype)
        host[0::2] = arr.real
        host[1::2] = arr.imag
        cl.enqueue_copy(mon.fdtd.queue, buf, host)

    upload(mon.Ex_dft_buf, Ex)
    upload(mon.Ey_dft_buf, Ey)
    upload(mon.Ez_dft_buf, Ez)
    upload(mon.Hx_dft_buf, Hx)
    upload(mon.Hy_dft_buf, Hy)
    upload(mon.Hz_dft_buf, Hz)
    mon.fdtd.queue.finish()


class TestFarfieldSyntheticPlaneWave(unittest.TestCase):
    """Ground-truth reconstruction check, no FDTD solve involved."""

    def _make_monitor(self):
        ensure_pyopencl_ctx()
        shape = (80, 80, 80)
        dl = 1.0e-3
        fdtd = OpenCLFDTD(shape, dl, npml=8, dtype=np.float32)
        fdtd.set_epsilon(np.ones(shape, dtype=np.float32))
        freq = 10e9
        # Box centered in the domain (corner-origin metres, matching
        # fdtd_lib.runner._mm_center_to_solver_m's convention): grid is
        # 80mm, so center = 40mm; 30mm box keeps ample PML clearance.
        mon = OpenCLNear2FarMonitor(fdtd, (0.040, 0.040, 0.040), (0.030, 0.030, 0.030), freq)
        k = 2 * np.pi * freq / C0
        return mon, dl, k

    def _peak_angle_deg(self, mon, dl, k, theta_true_deg):
        _fill_analytic_plane_wave(mon, dl, k, theta_true_deg)
        angles = np.linspace(-180.0, 180.0, 73)
        rad = np.deg2rad(angles)
        pts = np.column_stack([10.0 * np.sin(rad), np.zeros_like(rad), 10.0 * np.cos(rad)])
        eh = mon.get_farfields(pts)
        E, H = eh[:, 0:3], eh[:, 3:6]
        S = 0.5 * (E[:, 1] * np.conj(H[:, 2]) - E[:, 2] * np.conj(H[:, 1]))
        Sy = 0.5 * (E[:, 2] * np.conj(H[:, 0]) - E[:, 0] * np.conj(H[:, 2]))
        Sz = 0.5 * (E[:, 0] * np.conj(H[:, 1]) - E[:, 1] * np.conj(H[:, 0]))
        mag = np.sqrt(np.abs(S) ** 2 + np.abs(Sy) ** 2 + np.abs(Sz) ** 2)
        return float(angles[int(np.argmax(mag))])

    @unittest.expectedFailure
    def test_closed_box_reconstructs_peak_at_true_angle_0deg(self):
        mon, dl, k = self._make_monitor()
        peak = self._peak_angle_deg(mon, dl, k, 0.0)
        self.assertLess(abs(peak - 0.0), 10.0, f"expected peak near 0deg, got {peak}deg")

    @unittest.expectedFailure
    def test_closed_box_reconstructs_peak_at_true_angle_20deg(self):
        mon, dl, k = self._make_monitor()
        peak = self._peak_angle_deg(mon, dl, k, 20.0)
        self.assertLess(abs(peak - 20.0), 10.0, f"expected peak near 20deg, got {peak}deg")

    @unittest.expectedFailure
    def test_closed_box_reconstructs_peak_at_true_angle_40deg(self):
        mon, dl, k = self._make_monitor()
        peak = self._peak_angle_deg(mon, dl, k, 40.0)
        self.assertLess(abs(peak - 40.0), 10.0, f"expected peak near 40deg, got {peak}deg")


if __name__ == "__main__":
    unittest.main()
