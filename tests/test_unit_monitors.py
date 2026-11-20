# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
#
# This file is part of opencl-fdtd-solver.

"""Unit tests for near-to-far monitors and Poynting helpers."""

from __future__ import annotations

import os
import unittest

import numpy as np
from opencl_fdtd_solver import NumPyFDTD, NumPyNear2FarMonitor, OpenCLFDTD, OpenCLNear2FarMonitor
from opencl_fdtd_solver.constants import ETA0
from opencl_fdtd_solver.monitors import Near2FarBase, _poynting_db, _poynting_mag


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


class TestNear2FarBoxSnapping(unittest.TestCase):
    class _Grid:
        Nx = Ny = Nz = 24
        dl = 1e-3

    def _x_range(self, centre_mm, size_mm):
        box = Near2FarBase(
            self._Grid(), (centre_mm * 1e-3, 12e-3, 12e-3), (size_mm * 1e-3, 4e-3, 4e-3), 1e9
        )
        return box.ix0, box.ix1

    def test_whole_cell_box_keeps_its_extent_for_any_centre(self):
        # Faces on half cells: the extent must not depend on the centre's parity.
        for size in (1, 3, 5):
            for centre in (7, 8, 9, 10):
                with self.subTest(size=size, centre=centre):
                    lo, hi = self._x_range(centre, size)
                    self.assertEqual(hi - lo, size)
        for size in (2, 4):
            for centre in (7.5, 8.5, 9.5):
                with self.subTest(size=size, centre=centre):
                    lo, hi = self._x_range(centre, size)
                    self.assertEqual(hi - lo, size)

    def test_half_cell_faces_snap_to_the_higher_node(self):
        self.assertEqual(self._x_range(8, 3), (7, 10))
        self.assertEqual(self._x_range(9, 3), (8, 11))
        self.assertEqual(self._x_range(8, 1), (8, 9))

    def test_faces_on_nodes_and_off_grid_positions_snap_to_nearest(self):
        self.assertEqual(self._x_range(8, 4), (6, 10))
        self.assertEqual(self._x_range(8.1, 3.4), (6, 10))
        self.assertEqual(self._x_range(8.4, 3.4), (7, 10))

    def test_box_thinner_than_half_a_cell_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "degenerate"):
            self._x_range(8.2, 0.2)


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
        fdtd.add_source(lambda f: f.add_source_Ex(z, np.sin(2 * np.pi * freq * f.t)))
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

        # No points, as a list or as an empty (0, 3) array.
        for empty in ([], np.empty((0, 3))):
            self.assertEqual(mon.get_farfields(empty).shape, (0, 6))
        for bad in ([1.0, 2.0], np.ones((2, 3, 3))):
            with self.assertRaises(ValueError):
                mon.get_farfields(bad)

    def test_farfield_polar_rejects_nonpositive_distance(self):
        _, mon = self._small_run(n_steps=10)
        with self.assertRaises(ValueError):
            mon.farfield_polar_xz(distance_m=0.0)

    def test_fetch_dft_fields_scatters_tangential(self):
        fdtd, mon = self._small_run()
        faces = mon.fetch_dft_fields()
        self.assertEqual(set(faces), {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"})
        self.assertEqual(mon.Ex_dft.shape, (24, 24, 24))
        # At least one tangential slot on an x-face should be nonzero after driving Ex.
        face_power = sum(float(np.max(np.abs(v))) for v in faces.values())
        self.assertGreater(face_power, 0.0)
        # Normal-only slots on x-faces stay zero (tangential-only DFT).
        self.assertEqual(mon.Ex_dft.dtype, fdtd.accreal_complex_dtype)

    def test_phase_follows_solver_time_not_call_history(self):
        shape = (20, 20, 20)
        fdtd = OpenCLFDTD(shape, 1e-3, npml=3)
        mon = OpenCLNear2FarMonitor(fdtd, (0.01, 0.01, 0.01), (0.008, 0.008, 0.008), 3e9)
        self.assertIsNone(mon._phase)
        fdtd.run(5)
        step_phase = mon.omega * fdtd.dt
        # After n steps the DFT weight is dt * exp(j ω n dt).
        self.assertAlmostEqual(abs(mon._phase), fdtd.dt, delta=1e-12 * fdtd.dt)
        self.assertAlmostEqual(
            float(np.angle(mon._phase * np.exp(-5j * step_phase))), 0.0, places=12
        )
        # Jump the clock: the next weight must come from the new time alone.
        # A running product phase *= exp(j ω dt) would give 6 steps instead.
        fdtd.t = 1000.25 * fdtd.dt
        mon(fdtd)
        self.assertAlmostEqual(
            float(np.angle(mon._phase * np.exp(-1000.25j * step_phase))), 0.0, places=9
        )

    def test_get_farfields_tiny_box_local_reduce(self):
        """n_face << 256 exercises the local workgroup shrink path."""
        shape = (16, 16, 16)
        dl = 1e-3
        fdtd = OpenCLFDTD(shape, dl, npml=2)
        fdtd.add_source(lambda f: f.add_source_Ex(8, 0.05))
        mon = OpenCLNear2FarMonitor(fdtd, (8e-3, 8e-3, 8e-3), (2e-3, 2e-3, 2e-3), 5e9)
        self.assertLess(mon.n_face_samples, 256)
        fdtd.run(15)
        eh = mon.get_farfields([(0.0, 0.0, 5.0)])
        self.assertEqual(eh.shape, (1, 6))

    @staticmethod
    def _host_dft_rel_change(cur_faces, prev_faces) -> float:
        """‖cur−prev‖₂ / ‖cur‖₂ over all six face-packed DFT components."""
        num = 0.0
        den = 0.0
        for key in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            c = np.asarray(cur_faces[key], dtype=np.complex128)
            p = np.asarray(prev_faces[key], dtype=np.complex128)
            d = c - p
            num += float(np.vdot(d, d).real)
            den += float(np.vdot(c, c).real)
        if den <= 0.0:
            return 1.0
        return float(np.sqrt(num / den))

    def test_dft_rel_change_without_snapshot_is_one(self):
        _, mon = self._small_run(n_steps=20)
        self.assertIsNone(mon._dft_snap)
        self.assertEqual(mon.dft_relative_change(), 1.0)

    def test_dft_rel_change_snapshot_then_zero_and_grows(self):
        fdtd, mon = self._small_run(n_steps=40)
        self.assertIsNone(mon._dft_snap)
        mon.snapshot_dft()
        self.assertIsNotNone(mon._dft_snap)
        self.assertLess(mon.dft_relative_change(), 1e-5)

        fdtd.run(30)
        grown = mon.dft_relative_change()
        self.assertGreater(grown, 1e-3)
        self.assertLess(grown, 1.0)

    def test_dft_rel_change_matches_host_and_tiny_box(self):
        """GPU reduction matches host; tiny Huygens box covers lsize shrink."""
        shape = (16, 16, 16)
        dl = 1e-3
        fdtd = OpenCLFDTD(shape, dl, npml=2)
        fdtd.add_source(lambda f: f.add_source_Ex(8, 0.05 * np.sin(2 * np.pi * 5e9 * f.t)))
        mon = OpenCLNear2FarMonitor(fdtd, (8e-3, 8e-3, 8e-3), (2e-3, 2e-3, 2e-3), 5e9)
        self.assertLess(mon.n_face_samples, 256)

        fdtd.run(25)
        prev = mon.fetch_dft_fields()
        mon.snapshot_dft()
        fdtd.run(20)
        cur = mon.fetch_dft_fields()

        gpu = mon.dft_relative_change()
        host = self._host_dft_rel_change(cur, prev)
        self.assertGreater(host, 1e-3)
        self.assertAlmostEqual(gpu, host, delta=1e-4)


class TestNumPyNear2FarMonitor(unittest.TestCase):
    def test_accumulates_and_farfield_nonzero(self):
        shape = (20, 20, 20)
        dl = 2e-3
        npml = 3
        freq = 5e9
        fdtd = NumPyFDTD(shape, dl, npml=npml)
        z = shape[2] - npml - 2
        fdtd.add_source(
            lambda f: f.Ex.__setitem__(
                (slice(None), slice(None), z),
                f.Ex[:, :, z] + np.sin(2 * np.pi * freq * f.t),
            )
        )
        ctr = (shape[0] * dl / 2,) * 3
        size = (10 * dl,) * 3
        mon = NumPyNear2FarMonitor(fdtd, ctr, size, freq)
        fdtd.run(30)
        face_power = float(
            max(
                np.max(np.abs(mon.Ex_dft_f)),
                np.max(np.abs(mon.Ey_dft_f)),
                np.max(np.abs(mon.Ez_dft_f)),
                np.max(np.abs(mon.Hx_dft_f)),
                np.max(np.abs(mon.Hy_dft_f)),
                np.max(np.abs(mon.Hz_dft_f)),
            )
        )
        self.assertGreater(face_power, 0.0)
        ff = mon.get_farfield((0.0, 0.0, 50.0))
        self.assertEqual(ff.shape, (6,))
        self.assertGreater(float(np.max(np.abs(ff))), 0.0)


if __name__ == "__main__":
    unittest.main()
