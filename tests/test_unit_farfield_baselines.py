# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
#
# This file is part of opencl-fdtd-solver.

"""Unit tests for far-field golden helpers (no Meep required)."""

from __future__ import annotations

import unittest

import numpy as np

from tests.meep_validation.farfield_metrics import measure_farfield_case, null_ratio
from tests.meep_validation.farfield_render import compose_pattern_overlay


class TestFarfieldMetrics(unittest.TestCase):
    def test_null_ratio_and_measure(self):
        main = np.array([1, 0, 0, 0, 0, 0], dtype=np.complex128)
        null = np.array([0.01, 0, 0, 0, 0, 0], dtype=np.complex128)
        self.assertAlmostEqual(null_ratio(main, null), 0.01, places=12)
        angles = np.linspace(-180, 180, 5)
        ocl_db = np.array([0.0, -3.0, -10.0, -3.0, 0.0])
        meep_db = ocl_db + 0.5
        ocl = {"eh_plus_z": main, "eh_plus_x": null}
        meep = {"eh_plus_z": main, "eh_plus_x": null * 0.5}
        m = measure_farfield_case(ocl_db, meep_db, ocl, meep)
        self.assertIn("main_lobe_max_abs_db_error", m)
        self.assertLess(m["main_lobe_max_abs_db_error"], 1.0)


class TestFarfieldRender(unittest.TestCase):
    def test_overlay_shape(self):
        angles = np.linspace(-180, 180, 19)
        ocl = -np.abs(angles) / 10.0
        meep = ocl - 0.2
        rgb = compose_pattern_overlay(angles, ocl, meep)
        self.assertEqual(rgb.shape[2], 3)
        self.assertEqual(rgb.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
