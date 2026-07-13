# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
#
# This file is part of opencl-fdtd-solver.

"""Unit tests for OpenCLFDTD engine utilities and edge cases."""

from __future__ import annotations

import io
import os
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import pyopencl as cl

from opencl_fdtd_solver import OpenCLFDTD, NumPyFDTD


class TestMemoryEstimate(unittest.TestCase):
    def test_fields_only_when_npml_zero(self):
        shape = (40, 50, 60)
        item = 4
        expected = 7 * 40 * 50 * 60 * item
        self.assertEqual(
            OpenCLFDTD.estimate_device_memory_bytes(shape, npml=0),
            expected,
        )

    def test_includes_face_local_psi(self):
        shape = (100, 100, 100)
        npml = 10
        fields = 7 * 100 ** 3 * 4
        psi = (
            4 * (2 * npml * 100 * 100)
            + 4 * (100 * 2 * npml * 100)
            + 4 * (100 * 100 * 2 * npml)
        ) * 4
        self.assertEqual(
            OpenCLFDTD.estimate_device_memory_bytes(shape, npml),
            fields + psi,
        )

    def test_budget_reserves_headroom(self):
        class _Dev:
            global_mem_size = 16 * 1024 ** 3

        budget = OpenCLFDTD.device_memory_budget_bytes(_Dev())
        self.assertLess(budget, _Dev.global_mem_size)
        reserve = _Dev.global_mem_size - budget
        self.assertGreaterEqual(reserve, OpenCLFDTD.MEMORY_HEADROOM_BYTES)
        self.assertGreaterEqual(
            reserve, int(_Dev.global_mem_size * OpenCLFDTD.MEMORY_HEADROOM_FRACTION)
        )


class TestOpenCLEngineBasics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("PYOPENCL_CTX", "0")

    def test_courant_dt_and_progress(self):
        fdtd = OpenCLFDTD((16, 16, 16), 1e-3, npml=4)
        expected_dt = 0.99 * 1e-3 / (299_792_458.0 * np.sqrt(3.0))
        self.assertAlmostEqual(fdtd.dt, expected_dt, places=16)

        buf = io.StringIO()
        with redirect_stdout(buf):
            fdtd.run(3, progress_every=1)
        out = buf.getvalue()
        self.assertIn("step 0/3", out)
        self.assertEqual(fdtd.step_num, 3)

    def test_set_epsilon_roundtrip(self):
        fdtd = OpenCLFDTD((12, 12, 12), 1e-3, npml=2)
        eps = np.ones((12, 12, 12), dtype=np.float32)
        eps[6, 6, 6] = 4.0
        fdtd.set_epsilon(eps)
        host = np.empty(12 * 12 * 12, dtype=np.float32)
        cl.enqueue_copy(fdtd.queue, host, fdtd.eps_buf)
        fdtd.queue.finish()
        self.assertAlmostEqual(float(host.reshape(12, 12, 12)[6, 6, 6]), 4.0)

    def test_set_epsilon_shape_mismatch(self):
        fdtd = OpenCLFDTD((10, 10, 10), 1e-3, npml=2)
        with self.assertRaises(AssertionError):
            fdtd.set_epsilon(np.ones((8, 8, 8), dtype=np.float32))

    def test_npml_zero_runs(self):
        fdtd = OpenCLFDTD((20, 20, 20), 1e-3, npml=0)
        self.assertEqual(fdtd.psi_x_size, 0)
        fdtd._sources.append(lambda f: f.add_source_Ex(10, 0.1))
        fdtd.run(5)
        self.assertGreater(float(np.max(np.abs(fdtd.Ex))), 0.0)

    def test_memory_error_when_over_budget(self):
        tiny = mock.Mock()
        tiny.global_mem_size = 64 * 1024 * 1024  # 64 MiB
        tiny.name = "MockTinyGPU"
        tiny.type = cl.device_type.GPU

        ctx = mock.Mock()
        ctx.devices = [tiny]
        queue = mock.Mock()

        with self.assertRaises(MemoryError) as cm:
            OpenCLFDTD((400, 400, 400), 1e-3, npml=20, ctx=ctx, queue=queue)
        self.assertIn("usable", str(cm.exception).lower())

    def test_field_properties_shapes(self):
        shape = (14, 15, 16)
        fdtd = OpenCLFDTD(shape, 1e-3, npml=3)
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            arr = getattr(fdtd, name)
            self.assertEqual(arr.shape, shape)
            self.assertEqual(arr.dtype, np.float32)


class TestNumPyEngine(unittest.TestCase):
    def test_matches_opencl_vacuum_no_dielectric(self):
        os.environ.setdefault("PYOPENCL_CTX", "0")
        shape = (24, 24, 24)
        dl = 2e-3
        npml = 4
        freq = 4e9
        np_sim = NumPyFDTD(shape, dl, npml=npml)
        cl_sim = OpenCLFDTD(shape, dl, npml=npml)
        z = shape[2] - npml - 2

        np_sim._sources.append(lambda f: f.Ex.__setitem__((slice(None), slice(None), z), f.Ex[:, :, z] + np.sin(2 * np.pi * freq * f.t)))
        cl_sim._sources.append(lambda f: f.add_source_Ex(z, np.sin(2 * np.pi * freq * f.t)))
        np_sim.run(20)
        cl_sim.run(20)
        for field in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            diff = np.max(np.abs(getattr(np_sim, field) - getattr(cl_sim, field)))
            self.assertLess(diff, 2e-4, field)

    def test_run_progress_prints(self):
        sim = NumPyFDTD((12, 12, 12), 1e-3, npml=2)
        buf = io.StringIO()
        with redirect_stdout(buf):
            sim.run(2, progress_every=1)
        self.assertIn("step 0/2", buf.getvalue())

    def test_monitor_callback_invoked(self):
        sim = NumPyFDTD((12, 12, 12), 1e-3, npml=2)
        hits = []
        sim._monitors.append(lambda f: hits.append(f.step_num))
        sim.run(3)
        self.assertEqual(hits, [1, 2, 3])


class TestDeviceSelectionFallbacks(unittest.TestCase):
    def test_prefers_gpu_then_cpu(self):
        gpu = mock.Mock()
        gpu.name = "FakeGPU"
        gpu.type = cl.device_type.GPU
        gpu.global_mem_size = 8 * 1024 ** 3

        cpu = mock.Mock()
        cpu.name = "FakeCPU"
        cpu.type = cl.device_type.CPU
        cpu.global_mem_size = 8 * 1024 ** 3

        plat = mock.Mock()
        plat.get_devices = mock.Mock(side_effect=lambda dtype=None: [gpu] if dtype == cl.device_type.GPU else [cpu])

        fake_ctx = mock.Mock()
        fake_ctx.devices = [gpu]
        fake_queue = mock.Mock()

        with mock.patch("opencl_fdtd_solver.engine.cl.get_platforms", return_value=[plat]), \
             mock.patch("opencl_fdtd_solver.engine.cl.Context", return_value=fake_ctx) as ctx_ctor, \
             mock.patch("opencl_fdtd_solver.engine.cl.CommandQueue", return_value=fake_queue), \
             mock.patch.object(OpenCLFDTD, "_check_device_memory"), \
             mock.patch.object(OpenCLFDTD, "_build_cpml"), \
             mock.patch.object(OpenCLFDTD, "_compile_kernels"), \
             mock.patch("opencl_fdtd_solver.engine.cl.Buffer", return_value=mock.Mock()), \
             mock.patch("opencl_fdtd_solver.engine.cl.enqueue_copy"):
            fdtd = OpenCLFDTD((8, 8, 8), 1e-3, npml=1)

        ctx_ctor.assert_called_once()
        self.assertIs(fdtd.device, gpu)


if __name__ == "__main__":
    unittest.main()
