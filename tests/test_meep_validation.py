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

"""
Extensive OpenCL ↔ MEEP validation for every currently supported feature.

Requires local `meep` or Docker image `local-pymeep:latest`. Failures are hard
errors (no soft-skip), so CI/local runs surface missing Meep instead of silent pass.
"""

from __future__ import annotations

import unittest

import numpy as np
from opencl_fdtd_solver.constants import ETA0
from tests.meep_validation import (
    MeepUnavailableError,
    complex_align,
    max_abs_db_error,
    peak_normalize,
)
from tests.meep_validation.cases import (
    _sphere_eps,
    eh_from_list,
    run_meep_farfield_pattern,
    run_meep_nearfield_dft,
    run_meep_pml_decay,
    run_opencl_farfield_pattern,
    run_opencl_nearfield_dft,
    run_opencl_pml_decay,
)


def _require_meep_callable():
    """Probe Meep once; skip only if explicitly allowed via env (not default)."""
    import os

    try:
        from tests.meep_validation import run_meep_script

        run_meep_script("import meep as mp\nprint('MEEP_JSON:{\"ok\": true}')\n")
    except MeepUnavailableError as e:
        if os.environ.get("ALLOW_SKIP_MEEP", "").strip() in ("1", "true", "yes"):
            raise unittest.SkipTest(str(e))
        raise AssertionError(
            f"Meep validation requires Meep (local or Docker). {e}\n"
            "Install pymeep, or build/run local-pymeep:latest. "
            "Set ALLOW_SKIP_MEEP=1 only for environments without Meep."
        ) from e


class TestMeepValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _require_meep_callable()

    def test_01_nearfield_ex_dft_vacuum(self):
        """Yee + CPML + Ex sheet: DFT(Ex) at interior probes vs Meep (peak-normalized)."""
        probes = [(15, 15, 15), (15, 15, 18), (12, 15, 15), (15, 12, 16)]
        cl = run_opencl_nearfield_dft(probes)
        mp = run_meep_nearfield_dft(probes)

        cl_mags = np.array(
            [abs(complex(p["Ex_dft_real"], p["Ex_dft_imag"])) for p in cl["probes"]],
            dtype=np.float64,
        )
        mp_mags = np.array(
            [abs(complex(p["Ex_dft_real"], p["Ex_dft_imag"])) for p in mp["probes"]],
            dtype=np.float64,
        )
        self.assertGreater(float(np.max(cl_mags)), 0.0)
        self.assertGreater(float(np.max(mp_mags)), 0.0)

        cl_n = peak_normalize(cl_mags)
        mp_n = peak_normalize(mp_mags)
        err = float(np.max(np.abs(cl_n - mp_n)))
        self.assertLess(
            err,
            0.25,
            f"Near-field DFT shape mismatch (peak-norm max err={err:.3f}).\n"
            f"OpenCL={cl_n}\nMeep={mp_n}",
        )

    def test_02_farfield_pattern_vacuum(self):
        """Near-to-far |S|(θ) on XZ cut vs Meep (peak-aligned dB, main lobe)."""
        n_ang = 19
        cl = run_opencl_farfield_pattern(n_angles=n_ang)
        mp = run_meep_farfield_pattern(n_angles=n_ang)

        cl_db = np.asarray(cl["S_db"], dtype=np.float64)
        mp_db = np.asarray(mp["S_db"], dtype=np.float64)
        self.assertEqual(len(cl_db), n_ang)
        # Deep null floors differ (grid dispersion / PML); compare within -12 dB of peak.
        err = max_abs_db_error(cl_db, mp_db, mask_db=-12.0)
        self.assertLess(
            err,
            2.5,
            f"Vacuum far-field main-lobe max peak-aligned error {err:.3f} dB > 2.5 dB",
        )

    def test_03_farfield_vector_components_vacuum(self):
        """Ex-source: Ex/Hy on +z main lobe; deep null on +x (endfire)."""
        cl = run_opencl_farfield_pattern(n_angles=9)
        mp = run_meep_farfield_pattern(n_angles=9)

        cl_z = eh_from_list(cl["eh_plus_z"])
        mp_z = eh_from_list(mp["eh_plus_z"])
        mp_z[3:6] = mp_z[3:6] / ETA0

        # Polarization only where the pattern is strong (+z for Ex drive).
        for tag, sl in (("E", slice(0, 3)), ("H", slice(3, 6))):
            a = np.asarray(cl_z[sl], dtype=np.complex128)
            b = np.asarray(mp_z[sl], dtype=np.complex128)
            i = int(np.argmax(np.abs(b)))
            self.assertGreater(abs(a[i]), 0.0)
            a = a * (abs(b[i]) / abs(a[i]))
            a = np.array([complex_align(ai, bi) for ai, bi in zip(a, b)])
            a_n = a / np.linalg.norm(a)
            b_n = b / np.linalg.norm(b)
            err = float(np.max(np.abs(a_n - b_n)))
            self.assertLess(
                err,
                0.35,
                f"Far-field {tag} mismatch on +z: max |Δ|={err:.3f}",
            )

        # +x is near the Ex-dipole null: |E| ≪ main lobe (noise; do not compare pol.).
        cl_x = eh_from_list(cl["eh_plus_x"])
        mp_x = eh_from_list(mp["eh_plus_x"])
        for name, main, null in (
            ("opencl", cl_z, cl_x),
            ("meep", mp_z, mp_x),
        ):
            r = float(np.linalg.norm(null[:3]) / (np.linalg.norm(main[:3]) + 1e-30))
            self.assertLess(r, 0.05, f"{name} |E(+x)|/|E(+z)|={r:.3e} (expected null)")

    def test_04_dielectric_sphere_farfield_pattern(self):
        """εᵣ=4 sphere (inside Huygens box): far-field pattern shape vs Meep (main lobe)."""
        from tests.meep_validation.cases import SPHERE_RAD_CELLS, SPHERE_RADIUS_MM

        n_ang = 19
        eps = _sphere_eps(4.0, rad_cells=SPHERE_RAD_CELLS)
        cl = run_opencl_farfield_pattern(n_angles=n_ang, eps=eps)
        mp = run_meep_farfield_pattern(
            n_angles=n_ang, eps_sphere=4.0, sphere_radius_mm=SPHERE_RADIUS_MM
        )

        err = max_abs_db_error(
            np.asarray(cl["S_db"], dtype=np.float64),
            np.asarray(mp["S_db"], dtype=np.float64),
            mask_db=-12.0,
        )
        self.assertLess(
            err,
            3.0,
            f"Dielectric-sphere far-field main-lobe max peak-aligned error {err:.3f} dB > 3 dB",
        )

    def test_05_pml_energy_decay(self):
        """CPML absorbs the pulse: late/peak energy ratio small and comparable to Meep."""
        cl = run_opencl_pml_decay()
        mp = run_meep_pml_decay()

        self.assertGreater(cl["energy_peak"], 0.0)
        self.assertGreater(mp["energy_peak"], 0.0)
        self.assertLess(
            cl["ratio_late_over_peak"],
            0.05,
            f"OpenCL PML weak: late/peak={cl['ratio_late_over_peak']:.3e}",
        )
        self.assertLess(
            mp["ratio_late_over_peak"],
            0.05,
            f"Meep PML weak: late/peak={mp['ratio_late_over_peak']:.3e}",
        )
        # Both already well absorbed; don't require matching absolute residual floors
        # (CPML vs Meep PML differ by orders of magnitude at round-off).


if __name__ == "__main__":
    unittest.main()
