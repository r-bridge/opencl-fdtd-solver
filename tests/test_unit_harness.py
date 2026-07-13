# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
#
# This file is part of opencl-fdtd-solver.

"""Unit tests for MEEP validation harness helpers (no Docker / Meep required)."""

from __future__ import annotations

import json
import os
import unittest

import numpy as np

from tests.meep_validation.harness import (
    complex_align,
    ensure_pyopencl_ctx,
    gaussian_sine_amp,
    max_abs_db_error,
    meep_until,
    opencl_dt,
    parse_meep_json,
    peak_normalize,
    poynting_db_from_eh,
    rel_mag_error,
)


class TestHarnessMath(unittest.TestCase):
    def test_opencl_dt_and_meep_until(self):
        dl = 2e-3
        self.assertGreater(opencl_dt(dl), 0.0)
        self.assertAlmostEqual(meep_until(100, dl), 100 * 0.99 / np.sqrt(3.0) * 2.0, places=10)

    def test_gaussian_sine_amp_peakish(self):
        freq = 5e9
        fw = 0.2 * freq
        amps = [gaussian_sine_amp(t, freq, fw) for t in np.linspace(0, 8 / (np.pi * fw), 200)]
        self.assertGreater(max(abs(a) for a in amps), 0.5)

    def test_peak_normalize(self):
        x = np.array([1.0, -3.0, 2.0])
        y = peak_normalize(x)
        self.assertAlmostEqual(float(np.max(np.abs(y))), 1.0)
        with self.assertRaises(ValueError):
            peak_normalize(np.zeros(3))

    def test_max_abs_db_error_mask(self):
        a = np.array([0.0, -5.0, -40.0])
        b = np.array([0.0, -6.0, -80.0])
        full = max_abs_db_error(a, b)
        masked = max_abs_db_error(a, b, mask_db=-10.0)
        self.assertGreater(full, masked)
        self.assertLess(masked, 1.5)

    def test_max_abs_db_error_empty_mask_falls_back(self):
        a = np.array([-50.0, -60.0])
        b = np.array([-50.0, -70.0])
        # peaks at -50; mask_db=-5 → nothing near peak
        err = max_abs_db_error(a, b, mask_db=-5.0)
        self.assertGreaterEqual(err, 0.0)

    def test_rel_mag_and_complex_align(self):
        self.assertLess(rel_mag_error(2 + 0j, 2.2 + 0j), 0.1)
        aligned = complex_align(1j, -1 + 0j)
        self.assertAlmostEqual(np.angle(aligned), np.angle(-1 + 0j), places=6)
        self.assertEqual(complex_align(0j, 1 + 0j), 0j)

    def test_poynting_db_from_eh(self):
        eh = np.array([1, 0, 0, 0, 1 / 377.0, 0], dtype=np.complex128)
        db, mag = poynting_db_from_eh(eh)
        self.assertGreater(mag, 0.0)
        self.assertTrue(np.isfinite(db))

    def test_ensure_pyopencl_ctx_default(self):
        os.environ.pop("PYOPENCL_CTX", None)
        ensure_pyopencl_ctx()
        self.assertEqual(os.environ["PYOPENCL_CTX"], "0")

    def test_parse_meep_json(self):
        payload = {"ok": True, "v": 1}
        text = "noise\nMEEP_JSON:" + json.dumps(payload) + "\n"
        self.assertEqual(parse_meep_json(text), payload)
        with self.assertRaises(ValueError):
            parse_meep_json("no marker here")


if __name__ == "__main__":
    unittest.main()
