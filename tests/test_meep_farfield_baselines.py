# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
#
# This file is part of opencl-fdtd-solver.

"""Golden near-to-far patterns + discrepancy report: must match committed baselines."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.meep_validation.farfield_cases import (
    default_farfield_cases,
    load_case_farfields_from_baselines,
    run_case_farfields,
)
from tests.meep_validation.farfield_metrics import (
    build_farfield_discrepancy_document,
    case_farfield_report_dict,
    discrepancy_farfield_markdown,
    measure_farfield_case,
)
from tests.meep_validation.farfield_render import read_png_rgb, write_pattern_overlay_png
from tests.meep_validation.harness import MeepUnavailableError, OPENCL_COURANT, max_abs_db_error
from tests.meep_validation.plane_cases import baselines_root


def _require_meep():
    try:
        from tests.meep_validation import run_meep_script

        run_meep_script("import meep as mp\nprint('MEEP_JSON:{\"ok\": true}')\n")
    except MeepUnavailableError as e:
        if os.environ.get("ALLOW_SKIP_MEEP", "").strip() in ("1", "true", "yes"):
            raise unittest.SkipTest(str(e))
        raise AssertionError(
            f"Meep far-field baselines require Meep (local or Docker). {e}\n"
            "Install pymeep, or build/run local-pymeep:latest. "
            "Set ALLOW_SKIP_MEEP=1 only for environments without Meep."
        ) from e


def _rebuilt_doc_from_baselines():
    root = baselines_root()
    case_docs = []
    for case in default_farfield_cases():
        ocl, meep = load_case_farfields_from_baselines(root / case.name)
        metrics = measure_farfield_case(ocl["S_db"], meep["S_db"], ocl, meep)
        meta = json.loads((root / case.name / "meta.json").read_text(encoding="utf-8"))
        case_docs.append(
            case_farfield_report_dict(
                name=case.name,
                shape=list(meta["shape"]),
                dl_m=meta["dl_m"],
                npml=meta["npml"],
                n_steps=meta["n_steps"],
                freq_hz=meta["freq_hz"],
                fwidth_hz=meta["fwidth_hz"],
                n_angles=meta["n_angles"],
                sphere_eps=meta["sphere_eps"],
                sphere_rad_cells=meta["sphere_rad_cells"],
                max_main_lobe_db=meta["max_main_lobe_db"],
                courant=float(OPENCL_COURANT),
                metrics=metrics,
                images=list(meta["files"]),
            )
        )
    return build_farfield_discrepancy_document(case_docs)


class TestMeepFarfieldBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _require_meep()
        cls.patterns = {}
        for case in default_farfield_cases():
            cls.patterns[case.name] = run_case_farfields(case)

    def test_committed_baselines_and_report_exist(self):
        root = baselines_root()
        self.assertTrue(root.is_dir(), f"missing baselines root {root}")
        for name in ("DISCREPANCY_REPORT_FARFIELD.md", "discrepancy_report_farfield.json"):
            path = root / name
            self.assertTrue(path.is_file(), f"missing {path}; run update_farfield_baselines")
        for case in default_farfield_cases():
            meta = root / case.name / "meta.json"
            self.assertTrue(meta.is_file(), f"missing {meta}")
            data = json.loads(meta.read_text(encoding="utf-8"))
            self.assertIn("discrepancy", data)
            for name in data["files"]:
                path = root / case.name / name
                self.assertTrue(path.is_file(), f"missing baseline PNG {path}")

    def test_patterns_match_committed_baselines(self):
        root = baselines_root()
        for case in default_farfield_cases():
            with self.subTest(case=case.name):
                ocl, meep = self.patterns[case.name]
                o_ref, m_ref = load_case_farfields_from_baselines(root / case.name)
                np.testing.assert_allclose(
                    ocl["angles_deg"],
                    o_ref["angles_deg"],
                    rtol=0,
                    atol=1e-12,
                    err_msg=f"{case.name} angles",
                )
                np.testing.assert_allclose(
                    ocl["S_db"],
                    o_ref["S_db"],
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"{case.name} OpenCL S_db",
                )
                np.testing.assert_allclose(
                    meep["S_db"],
                    m_ref["S_db"],
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"{case.name} Meep S_db",
                )
                for key in ("eh_plus_z", "eh_plus_x"):
                    np.testing.assert_allclose(
                        ocl[key],
                        o_ref[key],
                        rtol=1e-4,
                        atol=1e-8,
                        err_msg=f"{case.name} OpenCL {key}",
                    )
                    np.testing.assert_allclose(
                        meep[key],
                        m_ref[key],
                        rtol=1e-4,
                        atol=1e-8,
                        err_msg=f"{case.name} Meep {key}",
                    )
                baseline = root / case.name / "pattern_xz.png"
                with tempfile.TemporaryDirectory() as tmp:
                    actual_path = Path(tmp) / "pattern_xz.png"
                    write_pattern_overlay_png(
                        actual_path, ocl["angles_deg"], ocl["S_db"], meep["S_db"]
                    )
                    expected = read_png_rgb(baseline)
                    actual = read_png_rgb(actual_path)
                if not np.array_equal(expected, actual):
                    diff = np.abs(
                        expected.astype(np.int16) - actual.astype(np.int16)
                    )
                    raise AssertionError(
                        f"{case.name}: PNG pixels differ from baseline "
                        f"(max|Δ|={int(diff.max())}, mean|Δ|={float(diff.mean()):.3f}). "
                        "Refresh with:\n"
                        "  IGNORE_GPU=NVIDIA,AMD PYOPENCL_CTX=0 python -m "
                        "tests.meep_validation.update_farfield_baselines"
                    )

    def test_discrepancy_report_matches_committed(self):
        root = baselines_root()
        rebuilt = _rebuilt_doc_from_baselines()
        expected = json.loads(
            (root / "discrepancy_report_farfield.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            rebuilt,
            expected,
            "discrepancy_report_farfield.json is stale vs committed arrays. "
            "Refresh with update_farfield_baselines.",
        )
        expected_md = (root / "DISCREPANCY_REPORT_FARFIELD.md").read_text(encoding="utf-8")
        self.assertEqual(
            discrepancy_farfield_markdown(rebuilt),
            expected_md,
            "DISCREPANCY_REPORT_FARFIELD.md is stale. Refresh with update_farfield_baselines.",
        )

    def test_discrepancy_quality_gates(self):
        for case in default_farfield_cases():
            with self.subTest(case=case.name):
                ocl, meep = self.patterns[case.name]
                err = max_abs_db_error(ocl["S_db"], meep["S_db"], mask_db=-12.0)
                self.assertLess(
                    err,
                    case.max_main_lobe_db,
                    f"{case.name}: main-lobe |Δ|dB={err:.3f} exceeds gate "
                    f"{case.max_main_lobe_db}",
                )
                for tag, main, null in (
                    ("opencl", ocl["eh_plus_z"], ocl["eh_plus_x"]),
                    ("meep", meep["eh_plus_z"], meep["eh_plus_x"]),
                ):
                    r = float(
                        np.linalg.norm(null[:3]) / (np.linalg.norm(main[:3]) + 1e-30)
                    )
                    self.assertLess(
                        r,
                        0.05,
                        f"{case.name} {tag} |E(+x)|/|E(+z)|={r:.3e}",
                    )


if __name__ == "__main__":
    unittest.main()
