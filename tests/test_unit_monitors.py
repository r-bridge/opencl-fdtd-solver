# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
#
# This file is part of opencl-fdtd-solver.

"""Unit tests for near-to-far monitors and Poynting helpers."""

from __future__ import annotations

import os
import unittest

import numpy as np

from opencl_fdtd_solver import OpenCLFDTD, NumPyFDTD, NumPyNear2FarMonitor, OpenCLNear2FarMonitor
from opencl_fdtd_solver.monitors import Near2FarBase, _poynting_db, _poynting_mag, ETA0


class TestPoyntingHelpers(unittest.TestCase):
    def test_poynting_mag_known(self):
        # Ex / Hy traveling in +z: S = ½ Re{Ex Hy*} = 0.5
        ff = np.array([1 + 0j, 0, 0, 0, 1 / ETA0, 0], dtype=np.complex128)
        # With H = Ex/η for free-space TEM, S_z = 0.5 * Ex * conj(Hy) = 0.5 / η0
        mag = _poynting_mag(ff)
        self.assertGreater(mag, 0.0)
        db, mag2 = _poynting_db(ff)
        self.assertEqual(mag2, mag)
        self.assertTrue(np.isfinite(db))

    def test_poynting_db_zero_is_neg_inf(self):
        db, mag = _poynting_db(np.zeros(6, dtype=np.complex128))
        self.assertEqual(mag, 0.0)
        self.assertEqual(db, float("-inf"))


class TestNear2FarBaseErrors(unittest.TestCase):
    def test_get_farfield_without_dft_raises(self):
        class _Dummy:
            Nx = Ny = Nz = 10
            dl = 1e-3

        base = Near2FarBase(_Dummy(), (5e-3, 5e-3, 5e-3), (4e-3, 4e-3, 4e-3), 1e9)
        with self.assertRaises(RuntimeError):
            base.get_farfield((0.0, 0.0, 1.0))


class TestOpenCLNear2FarMonitor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("PYOPENCL_CTX", "0")

    def _small_run(self, n_steps=40):
        shape = (24, 24, 24)
        dl = 2e-3
        npml = 4
        freq = 5e9
        fdtd = OpenCLFDTD(shape, dl, npml=npml)
        z = shape[2] - npml - 2
        fdtd._sources.append(lambda f: f.add_source_Ex(z, np.sin(2 * np.pi * freq * f.t)))
        ctr = (shape[0] * dl / 2, shape[1] * dl / 2, shape[2] * dl / 2)
        size = (12 * dl, 12 * dl, 12 * dl)
        mon = OpenCLNear2FarMonitor(fdtd, ctr, size, freq)
        fdtd.run(n_steps)
        return fdtd, mon

    def test_get_farfields_shape_and_vector_shape_errors(self):
        _, mon = self._small_run()
        pts = np.array([[0.0, 0.0, 10.0], [10.0, 0.0, 0.0]], dtype=np.float32)
        eh = mon.get_farfields(pts)
        self.assertEqual(eh.shape, (2, 6))

        with self.assertRaises(ValueError):
            mon.get_farfields(np.ones((3, 2), dtype=np.float32))

        # 1-D obs point path
        eh1 = mon.get_farfields(np.array([0.0, 0.0, 10.0], dtype=np.float32))
        self.assertEqual(eh1.shape, (1, 6))

    def test_farfield_polar_rejects_nonpositive_distance(self):
        _, mon = self._small_run(n_steps=10)
        with self.assertRaises(ValueError):
            mon.farfield_polar_xz(distance_m=0.0)

    def test_fetch_dft_fields_scatters_tangential(self):
        _, mon = self._small_run()
        faces = mon.fetch_dft_fields()
        self.assertEqual(set(faces), {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"})
        self.assertEqual(mon.Ex_dft.shape, (24, 24, 24))
        # At least one tangential slot on an x-face should be nonzero after driving Ex.
        face_power = sum(float(np.max(np.abs(v))) for v in faces.values())
        self.assertGreater(face_power, 0.0)
        # Normal-only slots on x-faces may stay zero (tangential-only DFT).
        self.assertEqual(mon.Ex_dft.dtype, np.complex64)

    def test_phase_recurrence_stable_across_steps(self):
        shape = (20, 20, 20)
        fdtd = OpenCLFDTD(shape, 1e-3, npml=3)
        mon = OpenCLNear2FarMonitor(fdtd, (0.01, 0.01, 0.01), (0.008, 0.008, 0.008), 3e9)
        self.assertIsNone(mon._phase)
        fdtd.run(5)
        self.assertIsNotNone(mon._phase)
        phase_after = complex(mon._phase)
        expected = np.exp(1j * mon.omega * fdtd.t) * fdtd.dt
        self.assertAlmostEqual(abs(phase_after), abs(expected), places=5)
        self.assertAlmostEqual(
            float(np.angle(phase_after / expected)),
            0.0,
            places=4,
        )

    def test_get_farfields_tiny_box_local_reduce(self):
        """n_face << 256 exercises the local workgroup shrink path."""
        shape = (16, 16, 16)
        dl = 1e-3
        fdtd = OpenCLFDTD(shape, dl, npml=2)
        fdtd._sources.append(lambda f: f.add_source_Ex(8, 0.05))
        mon = OpenCLNear2FarMonitor(fdtd, (8e-3, 8e-3, 8e-3), (2e-3, 2e-3, 2e-3), 5e9)
        self.assertLess(mon.n_face_samples, 256)
        fdtd.run(15)
        eh = mon.get_farfields([(0.0, 0.0, 5.0)])
        self.assertEqual(eh.shape, (1, 6))


class TestNumPyNear2FarMonitor(unittest.TestCase):
    def test_accumulates_and_farfield_nonzero(self):
        shape = (20, 20, 20)
        dl = 2e-3
        npml = 3
        freq = 5e9
        fdtd = NumPyFDTD(shape, dl, npml=npml)
        z = shape[2] - npml - 2
        fdtd._sources.append(
            lambda f: f.Ex.__setitem__(
                (slice(None), slice(None), z),
                f.Ex[:, :, z] + np.sin(2 * np.pi * freq * f.t),
            )
        )
        ctr = (shape[0] * dl / 2,) * 3
        size = (10 * dl,) * 3
        mon = NumPyNear2FarMonitor(fdtd, ctr, size, freq)
        fdtd.run(30)
        self.assertGreater(float(np.max(np.abs(mon.Ex_dft))), 0.0)
        ff = mon.get_farfield((0.0, 0.0, 50.0))
        self.assertEqual(ff.shape, (6,))
        self.assertGreater(float(np.max(np.abs(ff))), 0.0)


if __name__ == "__main__":
    unittest.main()
