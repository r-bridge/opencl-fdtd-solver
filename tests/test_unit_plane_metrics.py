# Copyright (C) 2026: OpenCL FDTD Solver Contributors
#
# This file is part of opencl-fdtd-solver.

"""Unit tests for discrepancy metrics (no Meep / Docker)."""

from __future__ import annotations

import unittest

import numpy as np

from tests.meep_validation.plane_metrics import (
    build_discrepancy_document,
    case_report_dict,
    discrepancy_markdown,
    measure_checkpoint,
)


class TestPlaneMetrics(unittest.TestCase):
    def test_identical_fields(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(20, 24))
        row = measure_checkpoint(a, a, npml=2, step=10)
        self.assertAlmostEqual(row.pearson_corr, 1.0, places=6)
        self.assertAlmostEqual(row.lms_scale, 1.0, places=6)
        self.assertAlmostEqual(row.raw_residual_energy_ratio, 0.0, places=12)
        self.assertEqual(row.mid_x_lag_cells, 0)

    def test_scaled_fields(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(16, 16))
        row = measure_checkpoint(a, a / 2.0, npml=1, step=1)
        self.assertAlmostEqual(row.pearson_corr, 1.0, places=6)
        self.assertAlmostEqual(row.lms_scale, 2.0, places=5)

    def test_markdown_contains_tables(self):
        row = measure_checkpoint(
            np.ones((8, 8)), np.ones((8, 8)), npml=1, step=5
        )
        case = case_report_dict(
            name="toy",
            shape=[8, 8, 8],
            dl_m=1e-3,
            npml=1,
            n_steps=5,
            freq_hz=1e9,
            fwidth_hz=2e8,
            block_half=[0, 0, 0],
            block_eps=1.0,
            courant=0.5,
            rows=[row],
            images=["step_0005.png"],
        )
        doc = build_discrepancy_document([case])
        md = discrepancy_markdown(doc)
        self.assertIn("# OpenCL ↔ Meep mid-plane Ex discrepancy report", md)
        self.assertIn("`toy`", md)
        self.assertIn("pearson_corr", md)


if __name__ == "__main__":
    unittest.main()
