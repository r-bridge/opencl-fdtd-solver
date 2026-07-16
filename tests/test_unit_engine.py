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
from opencl_fdtd_solver import NumPyFDTD, OpenCLFDTD
from opencl_fdtd_solver.kernels import KERNEL_FILES, load_kernel_source

EXPECTED_KERNELS = (
    "update_H_interior",
    "update_H_pml",
    "update_E_interior",
    "update_E_pml",
    "add_source_Ex",
    "add_source_Jx",
    "accumulate_dft",
    "accumulate_dft_face",
    "accumulate_dft_faces_fused",
    "dft_rel_change_partial",
    "farfield_accumulate_nl",
    "farfield_nl_to_eh",
)


class TestPackageMetadata(unittest.TestCase):
    def test_package_name_not_overwritten(self):
        import opencl_fdtd_solver

        self.assertEqual(opencl_fdtd_solver.__name__, "opencl_fdtd_solver")
        self.assertEqual(opencl_fdtd_solver.__version__, "1.0.0")


class TestKernelSources(unittest.TestCase):
    def test_packaged_cl_files_exist_and_list_kernels(self):
        src = load_kernel_source()
        self.assertEqual(KERNEL_FILES, ("yee_update.cl", "sources.cl", "dft_farfield.cl"))
        for name in EXPECTED_KERNELS:
            self.assertIn(f"__kernel void {name}", src, name)
        # Macro line-continuations must be single backslashes (not Python-escaped).
        self.assertRegex(src, r"#define DFT_ACC\(CUR, PREV\) \\\n")

    def test_opencl_program_builds(self):
        os.environ.setdefault("PYOPENCL_CTX", "0")
        ctx = cl.create_some_context(interactive=False)
        program = cl.Program(ctx, load_kernel_source()).build()
        for name in EXPECTED_KERNELS:
            cl.Kernel(program, name)


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
        fields = 7 * 100**3 * 4
        psi = (
            4 * (2 * npml * 100 * 100) + 4 * (100 * 2 * npml * 100) + 4 * (100 * 100 * 2 * npml)
        ) * 4
        self.assertEqual(
            OpenCLFDTD.estimate_device_memory_bytes(shape, npml),
            fields + psi,
        )

    def test_budget_reserves_headroom(self):
        class _Dev:
            global_mem_size = 16 * 1024**3

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
        with self.assertRaises(ValueError) as cm:
            fdtd.set_epsilon(np.ones((8, 8, 8), dtype=np.float32))
        self.assertIn("Epsilon shape mismatch", str(cm.exception))

    def test_npml_zero_runs(self):
        fdtd = OpenCLFDTD((20, 20, 20), 1e-3, npml=0)
        self.assertEqual(fdtd.psi_x_size, 0)
        fdtd.add_source(lambda f: f.add_source_Ex(10, 0.1))
        fdtd.run(5)
        fdtd.queue.finish()
        ex = fdtd.Ex
        self.assertTrue(np.all(np.isfinite(ex)))
        self.assertGreater(float(np.max(np.abs(ex))), 0.0)

    def test_add_source_jx_matches_soft_delta_e(self):
        """Jx inject equals Ex soft-add of -dt/(ε₀ εᵣ) J on the sheet."""
        from opencl_fdtd_solver.constants import EPS0

        shape = (16, 16, 16)
        dl = 1e-3
        npml = 2
        z = 8
        Jx = 0.25
        p = npml
        fdtd = OpenCLFDTD(shape, dl, npml=npml)
        eps = np.ones(shape, dtype=np.float32)
        eps[8, 8, z] = 4.0
        fdtd.set_epsilon(eps)
        fdtd.add_source(
            lambda f: f.add_source_Jx(z, Jx, i0=p, i1=shape[0] - p, j0=p, j1=shape[1] - p)
        )
        fdtd.step()
        fdtd.queue.finish()
        ex = fdtd.Ex
        soft_eps4 = -float(fdtd.dt) / (float(EPS0) * 4.0) * Jx
        soft_vac = -float(fdtd.dt) / float(EPS0) * Jx
        self.assertAlmostEqual(float(ex[8, 8, z]), soft_eps4, places=6)
        self.assertAlmostEqual(float(ex[p, p, z]), soft_vac, places=6)
        self.assertEqual(float(ex[0, 0, z]), 0.0)  # outside sheet (in PML)

    def test_add_source_jx_rim_taper_weights(self):
        """Rim taper: interior=1, edge=rim_edge, corner=rim_edge² (no renorm)."""
        from opencl_fdtd_solver.constants import EPS0

        shape = (16, 16, 16)
        npml = 2
        z = 8
        Jx = 1.0
        ew = 0.5
        p = npml
        i1 = shape[0] - p
        j1 = shape[1] - p
        fdtd = OpenCLFDTD(shape, 1e-3, npml=npml)
        fdtd.add_source(
            lambda f: f.add_source_Jx(
                z,
                Jx,
                i0=p,
                i1=i1,
                j0=p,
                j1=j1,
                rim_taper=True,
                rim_edge=ew,
                rim_renorm=False,
            )
        )
        fdtd.step()
        fdtd.queue.finish()
        ex = fdtd.Ex
        base = -float(fdtd.dt) / float(EPS0) * Jx
        self.assertAlmostEqual(float(ex[p + 2, p + 2, z]), base, places=6)  # interior
        self.assertAlmostEqual(float(ex[p, p + 2, z]), ew * base, places=6)  # x-edge
        self.assertAlmostEqual(float(ex[p + 2, p, z]), ew * base, places=6)  # y-edge
        self.assertAlmostEqual(float(ex[p, p, z]), ew * ew * base, places=6)  # corner
        self.assertAlmostEqual(float(ex[i1 - 1, j1 - 1, z]), ew * ew * base, places=6)

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

    def test_read_point_matches_field_slice(self):
        fdtd = OpenCLFDTD((12, 12, 12), 1e-3, npml=2)
        z = 5
        amp = 0.37
        fdtd.add_source(lambda f: f.add_source_Ex(z, amp))
        fdtd.step()
        i, j, k = 6, 7, z
        self.assertAlmostEqual(fdtd.read_point("Ex", i, j, k), float(fdtd.Ex[i, j, k]))

    def test_init_rejects_non_float32_dtype(self):
        gpu = mock.Mock()
        gpu.name = "FakeGPU"
        gpu.type = cl.device_type.GPU
        gpu.global_mem_size = 8 * 1024**3

        fake_ctx = mock.Mock()
        fake_ctx.devices = [gpu]
        fake_queue = mock.Mock()

        with self.assertRaises(ValueError) as cm:
            OpenCLFDTD(
                (8, 8, 8),
                1e-3,
                npml=1,
                dtype=np.float64,
                ctx=fake_ctx,
                queue=fake_queue,
            )
        self.assertIn("float32", str(cm.exception).lower())


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

        np_sim.add_source(lambda f: f.add_source_Ex(z, np.sin(2 * np.pi * freq * f.t)))
        cl_sim.add_source(lambda f: f.add_source_Ex(z, np.sin(2 * np.pi * freq * f.t)))
        np_sim.run(20)
        cl_sim.run(20)
        for field in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            diff = np.max(np.abs(getattr(np_sim, field) - getattr(cl_sim, field)))
            self.assertLess(diff, 2e-4, field)

    def test_set_epsilon_shape_mismatch(self):
        sim = NumPyFDTD((10, 10, 10), 1e-3, npml=2)
        with self.assertRaises(ValueError) as cm:
            sim.set_epsilon(np.ones((8, 8, 8), dtype=np.float32))
        self.assertIn("Epsilon shape mismatch", str(cm.exception))

    def test_add_source_jx_matches_soft_delta_e(self):
        """NumPy Jx inject equals Ex soft-add of -dt/(ε₀ εᵣ) J on the sheet."""
        from opencl_fdtd_solver.constants import EPS0

        shape = (16, 16, 16)
        dl = 1e-3
        npml = 2
        z = 8
        Jx = 0.25
        p = npml
        fdtd = NumPyFDTD(shape, dl, npml=npml)
        eps = np.ones(shape, dtype=np.float32)
        eps[8, 8, z] = 4.0
        fdtd.set_epsilon(eps)
        fdtd.add_source(
            lambda f: f.add_source_Jx(z, Jx, i0=p, i1=shape[0] - p, j0=p, j1=shape[1] - p)
        )
        fdtd.step()
        ex = fdtd.Ex
        soft_eps4 = -float(fdtd.dt) / (float(EPS0) * 4.0) * Jx
        soft_vac = -float(fdtd.dt) / float(EPS0) * Jx
        self.assertAlmostEqual(float(ex[8, 8, z]), soft_eps4, places=6)
        self.assertAlmostEqual(float(ex[p, p, z]), soft_vac, places=6)
        self.assertEqual(float(ex[0, 0, z]), 0.0)  # outside sheet (in PML)

    def test_add_source_jx_rim_taper_weights(self):
        """NumPy rim taper: interior=1, edge=rim_edge, corner=rim_edge² (no renorm)."""
        from opencl_fdtd_solver.constants import EPS0

        shape = (16, 16, 16)
        npml = 2
        z = 8
        Jx = 1.0
        ew = 0.5
        p = npml
        i1 = shape[0] - p
        j1 = shape[1] - p
        fdtd = NumPyFDTD(shape, 1e-3, npml=npml)
        fdtd.add_source(
            lambda f: f.add_source_Jx(
                z,
                Jx,
                i0=p,
                i1=i1,
                j0=p,
                j1=j1,
                rim_taper=True,
                rim_edge=ew,
                rim_renorm=False,
            )
        )
        fdtd.step()
        ex = fdtd.Ex
        base = -float(fdtd.dt) / float(EPS0) * Jx
        self.assertAlmostEqual(float(ex[p + 2, p + 2, z]), base, places=6)  # interior
        self.assertAlmostEqual(float(ex[p, p + 2, z]), ew * base, places=6)  # x-edge
        self.assertAlmostEqual(float(ex[p + 2, p, z]), ew * base, places=6)  # y-edge
        self.assertAlmostEqual(float(ex[p, p, z]), ew * ew * base, places=6)  # corner
        self.assertAlmostEqual(float(ex[i1 - 1, j1 - 1, z]), ew * ew * base, places=6)

    def test_add_source_jx_matches_opencl(self):
        """NumPy and OpenCL add_source_Jx (with rim taper + renorm) stay in lockstep."""
        os.environ.setdefault("PYOPENCL_CTX", "0")
        shape = (20, 20, 20)
        dl = 1e-3
        npml = 3
        z = 10
        p = npml
        Jx = 0.4
        eps = np.ones(shape, dtype=np.float32)
        eps[8:12, 8:12, z] = 2.5

        np_sim = NumPyFDTD(shape, dl, npml=npml)
        cl_sim = OpenCLFDTD(shape, dl, npml=npml)
        np_sim.set_epsilon(eps)
        cl_sim.set_epsilon(eps)

        def src(f):
            f.add_source_Jx(
                z,
                Jx,
                i0=p,
                i1=shape[0] - p,
                j0=p,
                j1=shape[1] - p,
                rim_taper=True,
                rim_edge=0.8,
                rim_renorm=True,
            )

        np_sim.add_source(src)
        cl_sim.add_source(src)
        np_sim.run(15)
        cl_sim.run(15)
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
        sim.add_monitor(lambda f: hits.append(f.step_num))
        sim.run(3)
        self.assertEqual(hits, [1, 2, 3])

    def test_add_source_rejects_non_callable(self):
        sim = NumPyFDTD((8, 8, 8), 1e-3, npml=1)
        with self.assertRaises(TypeError):
            sim.add_source(None)  # type: ignore[arg-type]

    def test_add_and_clear_sources(self):
        sim = NumPyFDTD((8, 8, 8), 1e-3, npml=1)
        hits = []
        sim.add_source(lambda f: hits.append("src"))
        sim.step()
        self.assertEqual(hits, ["src"])
        sim.clear_sources()
        sim.step()
        self.assertEqual(hits, ["src"])

    def test_add_and_clear_monitors(self):
        sim = NumPyFDTD((8, 8, 8), 1e-3, npml=1)
        hits = []
        sim.add_monitor(lambda f: hits.append("mon"))
        sim.step()
        self.assertEqual(hits, ["mon"])
        sim.clear_monitors()
        sim.step()
        self.assertEqual(hits, ["mon"])


class TestDeviceSelectionFallbacks(unittest.TestCase):
    def test_prefers_gpu_then_cpu(self):
        gpu = mock.Mock()
        gpu.name = "FakeGPU"
        gpu.type = cl.device_type.GPU
        gpu.global_mem_size = 8 * 1024**3

        fake_ctx = mock.Mock()
        fake_ctx.devices = [gpu]
        fake_queue = mock.Mock()

        with (
            mock.patch(
                "opencl_fdtd_solver.engine._default_opencl_runtime",
                return_value=(fake_ctx, fake_queue, gpu),
            ),
            mock.patch.object(OpenCLFDTD, "_check_device_memory"),
            mock.patch.object(OpenCLFDTD, "_build_cpml"),
            mock.patch.object(OpenCLFDTD, "_compile_kernels"),
            mock.patch("opencl_fdtd_solver.engine.cl.Buffer", return_value=mock.Mock()),
            mock.patch("opencl_fdtd_solver.engine.cl.enqueue_copy"),
        ):
            fdtd = OpenCLFDTD((8, 8, 8), 1e-3, npml=1)

        self.assertIs(fdtd.device, gpu)
        self.assertIs(fdtd.ctx, fake_ctx)
        self.assertIs(fdtd.queue, fake_queue)


if __name__ == "__main__":
    unittest.main()
