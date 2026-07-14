# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
#
# This file is part of opencl-fdtd-solver.

"""Golden mid-plane Ex images + discrepancy report: must match committed baselines."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.meep_validation.harness import MeepUnavailableError, OPENCL_COURANT
from tests.meep_validation.plane_cases import (
    baselines_root,
    default_plane_cases,
    load_case_planes_from_baselines,
    run_case_planes,
)
from tests.meep_validation.plane_metrics import (
    build_discrepancy_document,
    case_report_dict,
    discrepancy_markdown,
    measure_case,
)
from tests.meep_validation.plane_render import read_png_rgb, write_triptych_png


def _require_meep():
    try:
        from tests.meep_validation import run_meep_script

        run_meep_script("import meep as mp\nprint('MEEP_JSON:{\"ok\": true}')\n")
    except MeepUnavailableError as e:
        if os.environ.get("ALLOW_SKIP_MEEP", "").strip() in ("1", "true", "yes"):
            raise unittest.SkipTest(str(e))
        raise AssertionError(
            f"Meep plane baselines require Meep (local or Docker). {e}\n"
            "Install pymeep, or build/run local-pymeep:latest. "
            "Set ALLOW_SKIP_MEEP=1 only for environments without Meep."
        ) from e


def _live_discrepancy_doc(case_planes: dict[str, tuple[dict, dict]]):
    case_docs = []
    for case in default_plane_cases():
        ocl, meep = case_planes[case.name]
        rows = measure_case(ocl, meep, npml=case.npml, checkpoints=case.checkpoints)
        case_docs.append(
            case_report_dict(
                name=case.name,
                shape=list(case.shape),
                dl_m=case.dl,
                npml=case.npml,
                n_steps=case.n_steps,
                freq_hz=case.freq,
                fwidth_hz=case.fwidth,
                block_half=list(case.block_half),
                block_eps=case.block_eps,
                courant=float(OPENCL_COURANT),
                rows=rows,
                images=[f"step_{s:04d}.png" for s in case.checkpoints],
            )
        )
    return build_discrepancy_document(case_docs)


class TestMeepPlaneBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _require_meep()
        cls.planes = {}
        for case in default_plane_cases():
            cls.planes[case.name] = run_case_planes(case)

    def test_committed_baselines_and_report_exist(self):
        root = baselines_root()
        self.assertTrue(root.is_dir(), f"missing baselines root {root}")
        for name in ("DISCREPANCY_REPORT.md", "discrepancy_report.json"):
            path = root / name
            self.assertTrue(path.is_file(), f"missing {path}; run update_plane_baselines")
        for case in default_plane_cases():
            meta = root / case.name / "meta.json"
            self.assertTrue(meta.is_file(), f"missing {meta}")
            data = json.loads(meta.read_text(encoding="utf-8"))
            self.assertIn("discrepancy", data)
            for name in data["files"]:
                path = root / case.name / name
                self.assertTrue(path.is_file(), f"missing baseline PNG {path}")

    def test_planes_match_committed_baselines(self):
        """Live fields/PNGs must match git baselines."""
        root = baselines_root()
        for case in default_plane_cases():
            with self.subTest(case=case.name):
                ocl, meep = self.planes[case.name]
                for step in case.checkpoints:
                    o_ref = np.load(root / case.name / f"ocl_ex_step{step:04d}.npy")
                    m_ref = np.load(root / case.name / f"meep_ex_step{step:04d}.npy")
                    np.testing.assert_allclose(
                        ocl[step],
                        o_ref,
                        rtol=1e-5,
                        atol=1e-8,
                        err_msg=f"{case.name} OpenCL step {step}",
                    )
                    np.testing.assert_allclose(
                        meep[step],
                        m_ref,
                        rtol=1e-5,
                        atol=1e-8,
                        err_msg=f"{case.name} Meep step {step}",
                    )
                    baseline = root / case.name / f"step_{step:04d}.png"
                    with tempfile.TemporaryDirectory() as tmp:
                        actual_path = Path(tmp) / f"step_{step:04d}.png"
                        write_triptych_png(actual_path, ocl[step], meep[step])
                        expected = read_png_rgb(baseline)
                        actual = read_png_rgb(actual_path)
                    if not np.array_equal(expected, actual):
                        diff = np.abs(
                            expected.astype(np.int16) - actual.astype(np.int16)
                        )
                        raise AssertionError(
                            f"{case.name} step {step}: PNG pixels differ from "
                            f"baseline (max|Δ|={int(diff.max())}, "
                            f"mean|Δ|={float(diff.mean()):.3f}). "
                            "Refresh with:\n"
                            "  IGNORE_GPU=NVIDIA,AMD PYOPENCL_CTX=0 python -m "
                            "tests.meep_validation.update_plane_baselines"
                        )

    def test_discrepancy_report_matches_committed(self):
        """Committed report must match a fresh measure of the committed planes.

        Uses on-disk OpenCL/Meep arrays (not a live re-solve) so report identity
        is independent of OpenCL backend ULP noise (NVIDIA vs POCL).
        """
        root = baselines_root()
        case_docs = []
        for case in default_plane_cases():
            ocl, meep = load_case_planes_from_baselines(
                root / case.name, case.checkpoints
            )
            rows = measure_case(ocl, meep, npml=case.npml, checkpoints=case.checkpoints)
            case_docs.append(
                case_report_dict(
                    name=case.name,
                    shape=list(case.shape),
                    dl_m=case.dl,
                    npml=case.npml,
                    n_steps=case.n_steps,
                    freq_hz=case.freq,
                    fwidth_hz=case.fwidth,
                    block_half=list(case.block_half),
                    block_eps=case.block_eps,
                    courant=float(OPENCL_COURANT),
                    rows=rows,
                    images=[f"step_{s:04d}.png" for s in case.checkpoints],
                )
            )
        rebuilt = build_discrepancy_document(case_docs)
        expected = json.loads(
            (root / "discrepancy_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            rebuilt,
            expected,
            "discrepancy_report.json is stale vs committed planes. Refresh with "
            "update_plane_baselines.",
        )
        expected_md = (root / "DISCREPANCY_REPORT.md").read_text(encoding="utf-8")
        live_md = discrepancy_markdown(rebuilt)
        self.assertEqual(
            live_md,
            expected_md,
            "DISCREPANCY_REPORT.md is stale. Refresh with update_plane_baselines.",
        )

    def test_discrepancy_quality_gates(self):
        """Hard floors so silently-worsening agreement fails CI even if images refresh."""
        live = _live_discrepancy_doc(self.planes)
        for case in live["cases"]:
            with self.subTest(case=case["name"]):
                s = case["summary"]
                self.assertGreaterEqual(
                    s["mean_pearson_corr"],
                    0.95,
                    f"{case['name']}: mean Pearson corr too low",
                )
                self.assertLessEqual(
                    abs(s["mean_lms_scale"] - 1.0),
                    0.15,
                    f"{case['name']}: LMS scale too far from 1 (source mismatch)",
                )
                self.assertLessEqual(
                    s["mean_aligned_residual_energy_ratio"],
                    0.08,
                    f"{case['name']}: aligned residual energy too high",
                )
                self.assertLessEqual(
                    s["max_abs_mid_x_lag_cells"],
                    1,
                    f"{case['name']}: mid-x lag exceeds 1 cell",
                )


if __name__ == "__main__":
    unittest.main()
