# Copyright (C) 2026: OpenCL FDTD Solver Contributors
#
# This file is part of opencl-fdtd-solver.

"""Regression & physics tests for core numerical hardening fixes."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np
import pyopencl as cl
from opencl_fdtd_solver import (
    NumPyFDTD,
    NumPyFDTD_FaceCPML,
    NumPyNear2FarMonitor,
    OpenCLFDTD,
    OpenCLNear2FarMonitor,
)
from opencl_fdtd_solver.constants import ETA0
from opencl_fdtd_solver.monitors import Near2FarBase


class TestClosedBoxExtinctionTheorem(unittest.TestCase):
    """Source strictly outside a closed vacuum Huygens box must radiate ~0 far field."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("PYOPENCL_CTX", "0")

    def test_external_source_vanishes_on_closed_huygens_box(self):
        shape = (36, 36, 44)
        dl = 2e-3
        npml = 6
        freq = 5e9
        n_steps = 300

        t0 = 40.0
        sigma = 10.0

        def _run_case(box_center_cells):
            sim = OpenCLFDTD(shape, dl, npml=npml, dtype=np.float32)
            ctr = tuple(c * dl for c in box_center_cells)
            size = (10 * dl, 10 * dl, 10 * dl)
            gpu_mon = OpenCLNear2FarMonitor(sim, ctr, size, freq)
            cpu_mon = NumPyNear2FarMonitor(sim, ctr, size, freq)
            t0_s = t0 * sim.dt
            sigma_s = sigma * sim.dt

            def src(f):
                # Odd around t0_s -> zero DC integral (no leftover static charge).
                dt_t0 = f.t - t0_s
                env = np.exp(-0.5 * (dt_t0 / sigma_s) ** 2)
                amp = float(1e6 * env * np.sin(2.0 * np.pi * freq * dt_t0))
                f.add_source_Jx(14, amp, i0=17, i1=19, j0=17, j1=19)

            sim.add_source(src)
            sim.run(n_steps)
            obs = np.array(
                [
                    [0.0, 0.0, 10.0],
                    [0.0, 0.0, -10.0],
                    [10.0, 0.0, 0.0],
                    [0.0, 10.0, 0.0],
                    [6.0, 6.0, 6.0],
                ],
                dtype=np.float64,
            )
            e_gpu = gpu_mon.get_farfields(obs)[:, :3]
            e_cpu = np.array([cpu_mon.get_farfield(p)[:3] for p in obs])
            return float(np.max(np.linalg.norm(e_gpu, axis=1))), float(
                np.max(np.linalg.norm(e_cpu, axis=1))
            )

        # Source at z=14 is inside box [9..19], outside box [22..32].
        inside_gpu, inside_cpu = _run_case((18, 18, 14))
        outside_gpu, outside_cpu = _run_case((18, 18, 27))

        self.assertGreater(inside_gpu, 1e-12)
        self.assertGreater(inside_cpu, 1e-12)
        self.assertLess(
            outside_gpu / inside_gpu,
            2e-3,
            f"OpenCL extinction leak {outside_gpu / inside_gpu:.3e} too large",
        )
        self.assertLess(
            outside_cpu / inside_cpu,
            2e-3,
            f"NumPy extinction leak {outside_cpu / inside_cpu:.3e} too large",
        )


class TestAccumFp64AndFetchEdges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("PYOPENCL_CTX", "0")

    def test_accum_fp64_auto_enables_on_fp64_device_and_matches_fp32_accum(self):
        sim_auto = OpenCLFDTD((24, 24, 24), 2e-3, npml=4, dtype=np.float32)
        if OpenCLFDTD._device_supports_accum_fp64(sim_auto.device):
            self.assertTrue(sim_auto.accum_fp64)
            self.assertEqual(sim_auto.accreal, np.float64)
            self.assertEqual(sim_auto.accreal_complex_dtype, np.dtype(np.complex128))
        else:
            self.assertFalse(sim_auto.accum_fp64)

        sim_fp32_acc = OpenCLFDTD((24, 24, 24), 2e-3, npml=4, dtype=np.float32, accum_fp64=False)
        self.assertFalse(sim_fp32_acc.accum_fp64)
        self.assertEqual(sim_fp32_acc.accreal, np.float32)

        ctr = (0.024, 0.024, 0.024)
        size = (0.012, 0.012, 0.012)
        m1 = OpenCLNear2FarMonitor(sim_auto, ctr, size, 5e9)
        m2 = OpenCLNear2FarMonitor(sim_fp32_acc, ctr, size, 5e9)

        def src(f):
            f.add_source_Jx(12, float(np.sin(2.0 * np.pi * 5e9 * f.t)), i0=11, i1=13, j0=11, j1=13)

        sim_auto.add_source(src)
        sim_fp32_acc.add_source(src)
        sim_auto.run(60)
        sim_fp32_acc.run(60)

        # Compare against the peak, with H in E units: these fields are ~1e-14,
        # so an absolute tolerance would accept anything. At 1 m range the FP32
        # phase k*r is good to ~1e-5, well inside the gate.
        scale = np.array([1, 1, 1, ETA0, ETA0, ETA0])
        eh1 = m1.get_farfield((0.024, 0.024, 1.0)) * scale
        eh2 = m2.get_farfield((0.024, 0.024, 1.0)) * scale
        peak = float(np.max(np.abs(eh2)))
        self.assertGreater(peak, 0.0)
        self.assertLess(float(np.max(np.abs(eh1 - eh2))) / peak, 1e-4)

    def test_accum_fp64_auto_falls_back_on_fp32_only_device(self):
        dev = mock.Mock()
        dev.name = "FP32OnlyGPU"
        dev.type = cl.device_type.GPU
        dev.global_mem_size = 8 * 1024**3
        dev.extensions = "cl_khr_global_int32_base_atomics"
        ctx = mock.Mock()
        ctx.devices = [dev]
        queue = mock.Mock(context=ctx, device=dev)

        with (
            mock.patch.object(OpenCLFDTD, "_check_device_memory"),
            mock.patch.object(OpenCLFDTD, "_build_cpml"),
            mock.patch.object(OpenCLFDTD, "_compile_kernels"),
            mock.patch("opencl_fdtd_solver.engine.cl.Buffer", return_value=mock.Mock()),
            mock.patch("opencl_fdtd_solver.engine.cl.enqueue_copy"),
        ):
            sim = OpenCLFDTD((8, 8, 8), 1e-3, npml=1, ctx=ctx, queue=queue)
            self.assertFalse(sim.accum_fp64)
            self.assertEqual(sim.accreal, np.float32)

    def test_mixed_context_uses_queue_device_for_precision_and_build(self):
        fp32 = mock.Mock(extensions="", max_work_group_size=128)
        fp64 = mock.Mock(
            extensions="cl_khr_fp64 cl_khr_int64_base_atomics", max_work_group_size=128
        )
        ctx = mock.Mock(devices=[fp64, fp32])

        for dev, accum_fp64 in ((fp32, False), (fp64, True)):
            with (
                self.subTest(accum_fp64=accum_fp64),
                mock.patch.object(OpenCLFDTD, "_check_device_memory"),
                mock.patch.object(OpenCLFDTD, "_build_cpml"),
                mock.patch("opencl_fdtd_solver.engine.cl.Program") as program,
                mock.patch("opencl_fdtd_solver.engine.cl.Kernel") as kernel,
                mock.patch("opencl_fdtd_solver.engine.cl.Buffer"),
                mock.patch("opencl_fdtd_solver.engine.cl.enqueue_copy"),
            ):
                kernel.return_value.get_work_group_info.return_value = 128
                queue = mock.Mock(context=ctx, device=dev)
                sim = OpenCLFDTD((8, 8, 8), 1e-3, npml=1, queue=queue)
                self.assertIs(sim.ctx, ctx)
                self.assertIs(sim.device, dev)
                self.assertEqual(sim.accum_fp64, accum_fp64)
                build_args = program.return_value.build.call_args.kwargs
                self.assertEqual(build_args["devices"], [dev])
                self.assertIn(f"-DACCUM_FP64={int(accum_fp64)}", build_args["options"])

        other_ctx = mock.Mock(devices=[fp32])
        mismatched_queue = mock.Mock(context=other_ctx, device=fp32)
        with self.assertRaisesRegex(ValueError, "supplied context"):
            OpenCLFDTD((8, 8, 8), 1e-3, npml=1, ctx=ctx, queue=mismatched_queue)

        with (
            mock.patch.object(OpenCLFDTD, "_check_device_memory"),
            mock.patch.object(OpenCLFDTD, "_build_cpml"),
            mock.patch("opencl_fdtd_solver.engine.cl.CommandQueue") as queue_cls,
            mock.patch("opencl_fdtd_solver.engine.cl.Program") as program,
            mock.patch("opencl_fdtd_solver.engine.cl.Kernel") as kernel,
            mock.patch("opencl_fdtd_solver.engine.cl.Buffer"),
            mock.patch("opencl_fdtd_solver.engine.cl.enqueue_copy"),
        ):
            queue_cls.return_value = mock.Mock(context=ctx, device=fp32)
            kernel.return_value.get_work_group_info.return_value = 128
            sim = OpenCLFDTD((8, 8, 8), 1e-3, npml=1, ctx=ctx)
            queue_cls.assert_called_once_with(ctx)
            self.assertIs(sim.ctx, ctx)
            self.assertIs(sim.queue, queue_cls.return_value)
            self.assertIs(sim.device, fp32)
            self.assertFalse(sim.accum_fp64)
            self.assertEqual(program.return_value.build.call_args.kwargs["devices"], [fp32])

    def test_fetch_dft_fields_does_not_clobber_shared_edges_with_normal_zeros(self):
        sim = OpenCLFDTD((20, 20, 20), 1e-3, npml=3, dtype=np.float32)
        mon = OpenCLNear2FarMonitor(sim, (0.010,) * 3, (0.008,) * 3, 5e9)

        # Populate Ez on x/y faces with 1+2j while leaving z faces (normal) zero.
        ez = np.zeros(mon.n_face_samples, dtype=sim.accreal_complex_dtype)
        ez[mon._face_offsets[0] : mon._face_offsets[4]] = 1.0 + 2.0j
        cl.enqueue_copy(sim.queue, mon.Ez_dft_buf, ez)
        sim.queue.finish()

        mon.fetch_dft_fields()
        # Edge (ix0, iy0:iy1, iz0) belongs to x0 (tangential for Ez) and z0 (normal=0).
        edge_vals = mon.Ez_dft[mon.ix0, mon.iy0 : mon.iy1, mon.iz0]
        np.testing.assert_allclose(edge_vals, 1.0 + 2.0j)

    def test_fetch_dft_fields_host_farfield_keeps_every_face_sample(self):
        """Distinct samples on every face: the host integral must use all of them.

        Along the three box edges through the low corner two faces share a
        cell index, so a volume cannot hold both samples; the host path
        integrates the face packs instead.
        """
        sim = OpenCLFDTD((20, 22, 24), 1e-3, npml=3, dtype=np.float32)
        ctr, size, freq = (0.010, 0.011, 0.012), (0.008, 0.010, 0.006), 7e9
        gpu = OpenCLNear2FarMonitor(sim, ctr, size, freq)
        cpu = NumPyNear2FarMonitor(sim, ctr, size, freq)
        rng = np.random.default_rng(11)
        uploaded = {}
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            data = rng.normal(size=gpu.n_face_samples) + 1j * rng.normal(size=gpu.n_face_samples)
            data = data.astype(sim.accreal_complex_dtype)
            if name[0] == "H":
                data /= ETA0
            uploaded[name] = data
            cl.enqueue_copy(sim.queue, getattr(gpu, name + "_dft_buf"), data)
            getattr(cpu, name + "_dft_f")[:] = data
        sim.queue.finish()

        with self.assertRaises(RuntimeError):
            Near2FarBase.get_farfield(gpu, (0.0, 0.0, 1.0))
        faces = gpu.fetch_dft_fields()
        for name, data in uploaded.items():
            np.testing.assert_array_equal(faces[name], data)
            np.testing.assert_array_equal(getattr(gpu, name + "_dft_f"), data)

        scale = np.array([1, 1, 1, ETA0, ETA0, ETA0])
        for obs in ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.3, -0.4, 0.86)):
            with self.subTest(obs=obs):
                host = Near2FarBase.get_farfield(gpu, obs) * scale
                peak = float(np.max(np.abs(host)))
                self.assertGreater(peak, 0.0)
                device = gpu.get_farfield(obs) * scale
                reference = cpu.get_farfield(obs) * scale
                self.assertLess(float(np.max(np.abs(host - device))) / peak, 2e-5)
                self.assertLess(float(np.max(np.abs(host - reference))) / peak, 1e-12)

        # Volumes: face interiors are exact; on a shared low-corner edge the
        # later face (z after x) wins, as documented.
        x0 = uploaded["Ey"][: gpu._face_counts[0]].reshape(gpu.nyf, gpu.nzf)
        z0_start = gpu._face_offsets[4]
        z0 = uploaded["Ey"][z0_start : z0_start + gpu._face_counts[4]].reshape(gpu.nxf, gpu.nyf)
        np.testing.assert_array_equal(
            gpu.Ey_dft[gpu.ix0, gpu.iy0 : gpu.iy1, gpu.iz0 + 1 : gpu.iz1], x0[:, 1:]
        )
        np.testing.assert_array_equal(gpu.Ey_dft[gpu.ix0, gpu.iy0 : gpu.iy1, gpu.iz0], z0[0, :])

    def test_farfield_polar_xz_centered_on_box(self):
        sim = OpenCLFDTD((24, 24, 24), 2e-3, npml=4, dtype=np.float32)
        cuts = []
        for center in ((0.016, 0.016, 0.016), (0.028, 0.024, 0.028)):
            mon = OpenCLNear2FarMonitor(sim, center, (0.012,) * 3, 5e9)
            # Two tangential E samples give a nonconstant radiation pattern.
            ex = np.zeros(mon.n_face_samples, dtype=sim.accreal_complex_dtype)
            offset = mon._face_offsets[5]
            ex[offset] = 1
            ex[offset + mon.nyf] = 1j
            cl.enqueue_copy(sim.queue, mon.Ex_dft_buf, ex)
            angles, db = mon.farfield_polar_xz(distance_m=0.25, n_angles=9)
            np.testing.assert_allclose(angles, np.linspace(-180, 180, 9))
            self.assertTrue(np.all(np.isfinite(db)))
            self.assertGreater(np.ptp(db), 0.1)
            cuts.append(db)
        np.testing.assert_allclose(*cuts, rtol=0, atol=2e-5)

    def test_isotropic_tuple_of_planes_remains_supported(self):
        shape = (3, 4, 4)
        eps = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)
        for cls in (OpenCLFDTD, NumPyFDTD, NumPyFDTD_FaceCPML):
            with self.subTest(cls=cls.__name__):
                fields = []
                for material in (tuple(eps), eps):
                    sim = cls(shape, 1e-3, npml=0)
                    sim.set_epsilon(material)
                    sim.add_source_Jx(1, 2.0)
                    fields.append(np.asarray(sim.Ex))
                np.testing.assert_array_equal(*fields)

        sim = NumPyFDTD(shape, 1e-3, npml=0)
        ragged = ([[1.0, 2.0], [3.0]], np.ones(shape), np.ones(shape))
        with self.assertRaisesRegex(ValueError, r"Epsilon\[x\] must contain real numeric values"):
            sim.set_epsilon(ragged)


class TestAnisotropicLossyPecAndMixedPsiParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("PYOPENCL_CTX", "0")

    def test_numpy_mixed_float32_and_psi_float64_steps_cleanly(self):
        for cls in (NumPyFDTD, NumPyFDTD_FaceCPML):
            with self.subTest(cls=cls.__name__):
                sim = cls((16, 16, 16), 1e-3, npml=3, dtype=np.float32, psi_dtype=np.float64)
                sim.add_source(lambda f: f.add_source_Jx(8, 1.0, i0=6, i1=10, j0=6, j1=10))
                sim.run(10)
                self.assertEqual(sim.Ex.dtype, np.dtype(np.float32))
                self.assertEqual(sim._psi_Ex_y.dtype, np.dtype(np.float64))
                self.assertTrue(np.all(np.isfinite(sim.Ex)))

    def test_numpy_and_opencl_parity_for_lossy_and_pec(self):
        shape = (20, 20, 28)
        dl = 1e-3
        npml = 4
        eps = np.ones(shape, dtype=np.float32)
        eps[:, :, 16:20] = 2.5
        sigma = np.zeros(shape, dtype=np.float32)
        sigma[:, :, 16:20] = 15.0
        pec = np.zeros(shape, dtype=bool)
        pec[8:12, 8:12, 20] = True

        sims = [
            OpenCLFDTD(shape, dl, npml=npml, dtype=np.float32),
            NumPyFDTD(shape, dl, npml=npml, dtype=np.float32),
            NumPyFDTD_FaceCPML(shape, dl, npml=npml, dtype=np.float32),
        ]
        for sim in sims:
            sim.set_epsilon(eps, sigma_array=sigma, pec_mask=pec)
            sim.add_source(
                lambda f: f.add_source_Jx(
                    8, float(np.sin(2.0 * np.pi * 10e9 * f.t)), i0=6, i1=14, j0=6, j1=14
                )
            )
            sim.run(35)

        ex_cl = np.asarray(sims[0].Ex)
        for sim in sims[1:]:
            np.testing.assert_allclose(np.asarray(sim.Ex), ex_cl, rtol=1e-4, atol=1e-5)
            # PEC interior edges stay identically zero
            self.assertEqual(float(np.max(np.abs(np.asarray(sim.Ex)[8:12, 9:12, 20]))), 0.0)

    def test_diagonal_anisotropic_epsilon_affects_only_matching_polarization(self):
        shape = (18, 18, 30)
        dl = 1e-3
        npml = 4
        eps_hi = np.full(shape, 4.0, dtype=np.float32)
        eps_vac = np.ones(shape, dtype=np.float32)

        # Ex-polarized 1D plane wave propagating in z depends only on eps_x.
        cl_iso = OpenCLFDTD(shape, dl, npml=npml, dtype=np.float32)
        cl_aniso_x = OpenCLFDTD(shape, dl, npml=npml, dtype=np.float32)
        cl_aniso_yz = OpenCLFDTD(shape, dl, npml=npml, dtype=np.float32)
        np_aniso_x = NumPyFDTD(shape, dl, npml=npml, dtype=np.float32)

        cl_iso.set_epsilon(eps_hi)
        cl_aniso_x.set_epsilon((eps_hi, eps_vac, eps_vac))
        cl_aniso_yz.set_epsilon((eps_vac, eps_hi, eps_hi))
        np_aniso_x.set_epsilon((eps_hi, eps_vac, eps_vac))

        def src(f):
            f.add_source_Jx(10, float(np.sin(2.0 * np.pi * 8e9 * f.t)))

        for sim in (cl_iso, cl_aniso_x, cl_aniso_yz, np_aniso_x):
            sim.add_source(src)
            sim.run(30)

        # Center trace along z (pure Ex/Hy mode): (eps_hi, 1, 1) matches isotropic eps_hi,
        # while (1, eps_hi, eps_hi) differs dramatically (vacuum speed).
        ex_iso = np.asarray(cl_iso.Ex)[9, 9, :]
        ex_ax = np.asarray(cl_aniso_x.Ex)[9, 9, :]
        ex_ayz = np.asarray(cl_aniso_yz.Ex)[9, 9, :]
        ex_np = np.asarray(np_aniso_x.Ex)[9, 9, :]

        np.testing.assert_allclose(ex_ax, ex_iso, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(ex_np, ex_ax, rtol=1e-4, atol=1e-6)
        self.assertGreater(float(np.max(np.abs(ex_ayz - ex_iso))), 1e-3)

        # Validation rejects wrong tuple length or < 1 component.
        with self.assertRaisesRegex(ValueError, "must have 3 components"):
            cl_iso.set_epsilon((eps_vac, eps_vac))
        with self.assertRaises(ValueError):
            cl_iso.set_epsilon((eps_vac, np.full(shape, 0.5, dtype=np.float32), eps_vac))

    def test_tuple_of_wrong_shaped_planes_reports_the_shape(self):
        shape = (6, 4, 5)
        sim = NumPyFDTD(shape, 1e-3, npml=0)
        swapped = tuple(np.ones((5, 4)) for _ in range(6))
        too_few = tuple(np.ones((4, 5)) for _ in range(5))
        for planes, got in ((swapped, (6, 5, 4)), (too_few, (5, 4, 5))):
            with self.subTest(got=got):
                with self.assertRaises(ValueError) as cm:
                    sim.set_epsilon(planes)
                self.assertIn(
                    f"Epsilon shape mismatch: expected {shape}, got {got}", str(cm.exception)
                )


class TestBiaxialMediumAndPml(unittest.TestCase):
    """A biaxial medium must not reach the PML normal to its intermediate-ε axis."""

    SHAPE = (24, 24, 24)
    NPML = 6

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("PYOPENCL_CTX", "0")

    def _components(self, values, vacuum_pml_axes=""):
        comps = [np.full(self.SHAPE, v, dtype=np.float32) for v in values]
        for name in vacuum_pml_axes:
            axis = "xyz".index(name)
            for slab in (slice(0, self.NPML), slice(self.SHAPE[axis] - self.NPML, None)):
                index = tuple(slab if a == axis else slice(None) for a in range(3))
                for comp in comps:
                    comp[index] = 1.0
        return tuple(comps)

    def test_biaxial_in_intermediate_axis_pml_is_rejected(self):
        for cls in (OpenCLFDTD, NumPyFDTD, NumPyFDTD_FaceCPML):
            sim = cls(self.SHAPE, 1e-3, npml=self.NPML)
            # The intermediate component names the PML that must stay clear.
            for values, axis in (((2, 6, 1), "x"), ((6, 2, 1), "y"), ((6, 1, 2), "z")):
                others = "xyz".replace(axis, "")
                with self.subTest(cls=cls.__name__, values=values):
                    with self.assertRaisesRegex(ValueError, f"{axis}-normal PML"):
                        sim.set_epsilon(self._components(values))
                    with self.assertRaisesRegex(ValueError, f"{axis}-normal PML"):
                        sim.set_epsilon(self._components(values, vacuum_pml_axes=others))
                    sim.set_epsilon(self._components(values, vacuum_pml_axes=axis))

    def test_uniaxial_isotropic_and_pml_free_biaxial_are_accepted(self):
        for cls in (OpenCLFDTD, NumPyFDTD):
            with self.subTest(cls=cls.__name__):
                sim = cls(self.SHAPE, 1e-3, npml=self.NPML)
                for values in ((6, 2, 2), (6, 6, 1), (1, 6, 6), (3, 3, 3)):
                    sim.set_epsilon(self._components(values))
                # Integer and bool component arrays go through the same check.
                sim.set_epsilon(tuple(np.full(self.SHAPE, v, dtype=np.int64) for v in (4, 1, 1)))
                sim.set_epsilon(tuple(np.ones(self.SHAPE, dtype=bool) for _ in range(3)))
                no_pml = cls(self.SHAPE, 1e-3, npml=0)
                no_pml.set_epsilon(self._components((6, 2, 1)))

    def _field_growth(self, sim):
        """max|Ex| at step 3000 over max|Ex| at step 1500, long after the source ended."""
        centre = self.SHAPE[0] // 2

        def src(f):
            # Odd pulse (zero net charge) that ends at step 80.
            s = f.step_num - 40
            if abs(s) <= 40:
                amp = float(s / 10 * np.exp(-0.5 * (s / 10) ** 2))
                f.add_source_Jx(
                    centre, amp, i0=centre - 1, i1=centre + 1, j0=centre - 2, j1=centre + 1
                )

        sim.add_source(src)
        sim.run(1500)
        mid = float(np.max(np.abs(sim.Ex)))
        sim.run(1500)
        late = float(np.max(np.abs(sim.Ex)))
        self.assertGreater(mid, 0.0)
        return late / mid

    def test_biaxial_medium_clear_of_critical_pml_stays_stable(self):
        # Biaxial up to the x and z PMLs, vacuum only inside the y PML.
        sim = OpenCLFDTD(self.SHAPE, 1e-3, npml=self.NPML)
        sim.set_epsilon(self._components((6, 2, 1), vacuum_pml_axes="y"))
        growth = self._field_growth(sim)
        self.assertLess(growth, 2.0, f"field grew {growth:.2e}-fold after the source ended")

    def test_rejected_configuration_really_diverges(self):
        # With the check bypassed the same run grows about eightfold every 500
        # steps (measured growth here: ~470). This is what the rejection
        # protects against, and it shows that the run above is long enough to
        # notice an instability.
        sim = OpenCLFDTD(self.SHAPE, 1e-3, npml=self.NPML)
        with mock.patch("opencl_fdtd_solver._validation._reject_biaxial_in_pml"):
            sim.set_epsilon(self._components((6, 2, 1)))
        self.assertGreater(self._field_growth(sim), 10.0)


if __name__ == "__main__":
    unittest.main()
