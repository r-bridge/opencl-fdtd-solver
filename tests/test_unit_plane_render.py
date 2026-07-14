# Copyright (C) 2026: OpenCL FDTD Solver Contributors
#
# This file is part of opencl-fdtd-solver.

"""Unit tests for font-free plane PNG renderer (no Meep / Docker)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.meep_validation.plane_render import (
    compose_triptych,
    read_png_rgb,
    write_triptych_png,
)


class TestPlaneRender(unittest.TestCase):
    def test_roundtrip_deterministic(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(16, 20))
        b = 0.8 * a + 0.05 * rng.normal(size=a.shape)
        rgb1 = compose_triptych(a, b)
        rgb2 = compose_triptych(a, b)
        self.assertEqual(rgb1.dtype, np.uint8)
        self.assertEqual(rgb1.shape[2], 3)
        np.testing.assert_array_equal(rgb1, rgb2)
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "a.png"
            p2 = Path(tmp) / "b.png"
            write_triptych_png(p1, a, b)
            write_triptych_png(p2, a, b)
            np.testing.assert_array_equal(read_png_rgb(p1), read_png_rgb(p2))
            np.testing.assert_array_equal(read_png_rgb(p1), rgb1)


if __name__ == "__main__":
    unittest.main()
