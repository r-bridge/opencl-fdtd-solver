"""Independent regressions for the whole-project correctness audit."""

import unittest
from unittest import mock

import numpy as np
import pyopencl as cl
from opencl_fdtd_solver import (
    C0,
    EPS0,
    ETA0,
    NumPyFDTD,
    NumPyFDTD_FaceCPML,
    NumPyNear2FarMonitor,
    OpenCLFDTD,
    OpenCLNear2FarMonitor,
)
from opencl_fdtd_solver.monitors import _poynting_db
from tests.meep_validation.harness import max_abs_db_error, poynting_db_from_eh


class TestInputAndMemory(unittest.TestCase):
    def test_invalid_grids_rejected_before_allocation(self):
        for cls in (NumPyFDTD, NumPyFDTD_FaceCPML, OpenCLFDTD):
            for shape, dl, npml in (
                ((8, 8, 8), 1e-3, 5),
                ((8, 8, 8), 1e-3, -1),
                ((8, 8, 8), 1e-3, 1.5),
                ((8, 8, 8), 0, 2),
                ((8, 8, 8), np.nan, 2),
                ((0, 8, 8), 1e-3, 0),
                ((8, 8.5, 8), 1e-3, 0),
                ((8, 8), 1e-3, 0),
            ):
                with self.subTest(cls=cls.__name__, shape=shape, dl=dl, npml=npml):
                    with self.assertRaises(ValueError):
                        cls(shape, dl, npml=npml)

    def test_source_bounds_rejected_without_changing_fields(self):
        for cls in (NumPyFDTD, OpenCLFDTD):
            s = cls((8, 8, 8), 1e-3, npml=2)
            for method in (s.add_source_Ex, s.add_source_Jx):
                for z, bounds in (
                    (-1, {}),
                    (8, {}),
                    (1.5, {}),
                    (4, {"i0": -1}),
                    (4, {"i1": 9}),
                    (4, {"j0": 6, "j1": 3}),
                ):
                    with self.subTest(cls=cls.__name__, method=method.__name__, z=z, bounds=bounds):
                        with self.assertRaises(ValueError):
                            method(z, 1.0, **bounds)
            np.testing.assert_array_equal(s.Ex, 0)

    def test_kernel_rejects_invalid_z_even_without_host_validation(self):
        s = OpenCLFDTD((8, 8, 8), 1e-3, npml=2)
        for z in (-1, s.Nz):
            common = (
                s.queue,
                (s.Ny, s.Nx),
                None,
                *(np.int32(v) for v in (8, 8, 8, z)),
                s.real(1),
                *(np.int32(v) for v in (0, 8, 0, 8)),
            )
            s.kern_add_source_Ex(*common, s.Ex_buf)
            s.kern_add_source_Jx(*common, np.int32(0), s.real(0.8), s.ce_buf, s.Ex_buf)
        np.testing.assert_array_equal(s.Ex, 0)

    def test_estimate_matches_allocated_buffers(self):
        for npml in (0, 2):
            s = OpenCLFDTD((10, 12, 14), 1e-3, npml=npml)
            # Count actual allocations, deduplicating backward-compatible aliases.
            buffers = {v.int_ptr: v.size for v in vars(s).values() if isinstance(v, cl.Buffer)}
            self.assertEqual(
                sum(buffers.values()), s.estimate_device_memory_bytes((10, 12, 14), npml)
            )

    def test_large_grid_rejected_with_loss_coefficients_included(self):
        s = OpenCLFDTD.__new__(OpenCLFDTD)
        s.device = type("Device", (), {"global_mem_size": 12 * 1024**3, "name": "12 GiB"})()
        with self.assertRaises(MemoryError):
            s._check_device_memory((650, 650, 650), 20, np.float32)

    def test_unusable_platform_does_not_hide_working_device(self):
        import opencl_fdtd_solver.engine as engine

        broken = mock.Mock()
        broken.name = "unusable ICD"
        broken.get_devices.side_effect = cl.LogicError("device enumeration failed")
        device = mock.Mock()
        device.type = cl.device_type.CPU
        good = mock.Mock()
        good.get_devices.return_value = [device]
        with (
            mock.patch.multiple(
                engine, _DEFAULT_CTX=None, _DEFAULT_QUEUE=None, _DEFAULT_DEVICE=None
            ),
            mock.patch.dict("os.environ", {"IGNORE_GPU": ""}),
            mock.patch.object(cl, "get_platforms", return_value=[broken, good]),
            mock.patch.object(cl, "Context"),
            mock.patch.object(cl, "CommandQueue"),
        ):
            self.assertIs(engine._default_opencl_runtime()[2], device)


class TestSourceAndPrecision(unittest.TestCase):
    def test_conductive_drive_matches_uniform_ampere_recurrence(self):
        shape = (8, 8, 8)
        for loss_target in (0, 0.25, 1, 2):
            with self.subTest(loss=loss_target):
                s = OpenCLFDTD(shape, 1e-3, npml=2)
                sigma = np.full(shape, loss_target * 2 * EPS0 / s.dt, dtype=np.float32)
                s.set_epsilon(np.ones(shape), sigma_array=sigma)
                # Uniform drive keeps curl(H)=0, including the outer boundary.
                times = []

                def source(f):
                    times.append(f.t)
                    for z in range(f.Nz):
                        f.add_source_Jx(z, 1.0)

                s.add_source(source)
                loss = float(sigma[0, 0, 0]) * s.dt / (2 * EPS0)
                ca = (1 - loss) / (1 + loss)
                cb = s.dt / EPS0 / (1 + loss)
                expected = 0.0
                for _ in range(4):
                    s.step()
                    expected = ca * expected - cb
                    np.testing.assert_allclose(s.Ex, expected, rtol=2e-6, atol=1e-8)
                np.testing.assert_allclose(times, (np.arange(4) + 0.5) * s.dt, rtol=1e-14)
                s.clear_sources()
                s.step()
                np.testing.assert_allclose(s.Ex, ca * expected, rtol=2e-6, atol=1e-8)

    def test_degenerate_taper_parity_and_integrated_current(self):
        for nx, ny in ((1, 1), (1, 4), (4, 1), (2, 2), (4, 5)):
            for renorm in (False, True):
                with self.subTest(nx=nx, ny=ny, renorm=renorm):
                    a = NumPyFDTD((10, 10, 10), 1e-3, npml=2)
                    b = OpenCLFDTD((10, 10, 10), 1e-3, npml=2)
                    for s in (a, b):
                        s.add_source_Jx(
                            5,
                            1.0,
                            i0=2,
                            i1=2 + nx,
                            j0=2,
                            j1=2 + ny,
                            rim_taper=True,
                            rim_edge=0.5,
                            rim_renorm=renorm,
                        )
                    np.testing.assert_allclose(a.Ex, b.Ex, rtol=2e-6, atol=1e-8)
                    if renorm:
                        self.assertAlmostEqual(
                            float(np.sum(a.Ex)), -a.dt / EPS0 * nx * ny, places=5
                        )

    def test_zero_weight_renormalization_and_empty_sheets(self):
        for cls in (NumPyFDTD, OpenCLFDTD):
            s = cls((8, 8, 8), 1e-3, npml=2)
            with self.assertRaisesRegex(ValueError, "weights are all zero"):
                s.add_source_Jx(4, 1.0, i0=2, i1=3, rim_taper=True, rim_edge=0)
            s.add_source_Jx(4, 1.0, i0=2, i1=2, rim_taper=True, rim_edge=0)
            np.testing.assert_array_equal(s.Ex, 0)

    def test_psi_precision_and_allocations_persist_after_steps(self):
        for cls in (NumPyFDTD, NumPyFDTD_FaceCPML):
            s = cls((8, 8, 8), 1e-3, npml=2, psi_dtype=np.float16)
            arrays = {n: v for n, v in vars(s).items() if n.startswith("_psi_")}
            s.Ex[:] = np.random.default_rng(8).normal(size=s.Ex.shape) * 0.01
            s.run(3)
            for name, original in arrays.items():
                self.assertIs(getattr(s, name), original)
                self.assertEqual(getattr(s, name).dtype, np.float16)
            self.assertGreater(np.max(np.abs(s.Hy)), 0)


class TestFarfieldPhysics(unittest.TestCase):
    def test_radial_and_translation_phase_match_positive_time_dft(self):
        # With F(omega)=integral f(t) exp(+i*omega*t) dt, a delayed outgoing
        # wave gains exp(+i*k*distance). Moving its source toward the observer
        # shortens that delay and therefore contributes exp(-i*k*translation).
        s = OpenCLFDTD((16, 16, 16), 1e-3, npml=2)
        k = 2 * np.pi * 5e9 / C0
        for cls in (NumPyNear2FarMonitor, OpenCLNear2FarMonitor):
            values = []
            for shift in (0, 0.001):
                mon = cls(s, (0.008, 0.008, 0.008 + shift), (0.004,) * 3, 5e9)
                for name, value in (("Ex", 1.0), ("Hy", 1.0 / ETA0)):
                    data = np.zeros(mon.n_face_samples, dtype=np.complex64)
                    data[mon._face_offsets[5] :] = value
                    if isinstance(mon, NumPyNear2FarMonitor):
                        getattr(mon, name + "_dft_f")[:] = data
                    else:
                        cl.enqueue_copy(s.queue, getattr(mon, name + "_dft_buf"), data)
                a = mon.get_farfield((0, 0, 1.0))[0]
                b = mon.get_farfield((0, 0, 1.01))[0]
                np.testing.assert_allclose(b / a * 1.01, np.exp(1j * k * 0.01), rtol=2e-5)
                values.append(a)
            np.testing.assert_allclose(values[1] / values[0], np.exp(-1j * k * 0.001), rtol=2e-5)

    def test_huygens_face_direction_and_outward_flux(self):
        s = OpenCLFDTD((12, 12, 12), 1e-3, npml=2)
        cpu = NumPyNear2FarMonitor(s, (0.006,) * 3, (0.004,) * 3, 5e9)
        gpu = OpenCLNear2FarMonitor(s, (0.006,) * 3, (0.004,) * 3, 5e9)
        names = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        for axis in range(3):
            for sign in (-1, 1):
                with self.subTest(axis=axis, sign=sign):
                    direction = np.eye(3)[axis] * sign
                    e = np.eye(3)[(axis + 1) % 3]
                    h = np.cross(direction, e) / ETA0
                    face = 2 * axis + (sign > 0)
                    start = cpu._face_offsets[face]
                    count = cpu._face_counts[face]
                    for name, value in zip(names, np.concatenate((e, h))):
                        data = np.zeros(cpu.n_face_samples, dtype=np.complex64)
                        data[start : start + count] = value
                        getattr(cpu, name + "_dft_f")[:] = data
                        cl.enqueue_copy(s.queue, getattr(gpu, name + "_dft_buf"), data)
                    for mon in (cpu, gpu):
                        forward = mon.get_farfield(direction)
                        backward = mon.get_farfield(-direction)
                        amplitude = np.linalg.norm(forward[:3])
                        self.assertGreater(amplitude, 1e-6)
                        self.assertLess(np.linalg.norm(backward[:3]) / amplitude, 1e-5)
                        flux = 0.5 * np.real(np.cross(forward[:3], forward[3:].conj()))
                        self.assertGreater(float(np.dot(flux, direction)), 0)
                        np.testing.assert_allclose(
                            ETA0 * forward[3:],
                            np.cross(direction, forward[:3]),
                            rtol=1e-6,
                            atol=amplitude * 1e-7,
                        )
                    np.testing.assert_allclose(
                        cpu.get_farfield(direction),
                        gpu.get_farfield(direction),
                        rtol=2e-5,
                        atol=amplitude * 1e-7,
                    )

    def test_power_decibels(self):
        wave = np.array([1, 0, 0, 0, 1 / ETA0, 0], dtype=complex)
        for fn in (_poynting_db, poynting_db_from_eh):
            full, _ = fn(wave)
            half, _ = fn(wave / np.sqrt(2))
            self.assertAlmostEqual(half - full, -3.010299956639812, places=10)

    def test_missing_lobe_cannot_pass_pattern_gate(self):
        a = np.array([0.0, -3.0, -6.0, -40.0])
        b = np.array([0.0, -3.0, -40.0, -6.0])
        self.assertEqual(max_abs_db_error(a, b, mask_db=-12), 34.0)
