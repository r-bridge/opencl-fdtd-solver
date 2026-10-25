# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
#
# This file is part of opencl-fdtd-solver.
#
# opencl-fdtd-solver is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# opencl-fdtd-solver is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with opencl-fdtd-solver.  If not, see <http://www.gnu.org/licenses/>.

import time

import numpy as np
from opencl_fdtd_solver import NumPyFDTD, NumPyNear2FarMonitor, OpenCLFDTD, OpenCLNear2FarMonitor


def _benchmark_backend(
    solver_cls,
    monitor_cls,
    shape,
    dl,
    npml,
    z_src,
    source_bounds,
    freq,
    ctr_phys,
    size_phys,
    steps,
    warmup,
):
    sim = solver_cls(shape, dl, npml=npml)
    i0, i1, j0, j1 = source_bounds

    def source(fdtd):
        fdtd.add_source_Jx(
            z_src,
            np.sin(2 * np.pi * freq * fdtd.t),
            i0=i0,
            i1=i1,
            j0=j0,
            j1=j1,
        )

    sim.add_source(source)
    if monitor_cls is not None:
        monitor_cls(sim, ctr_phys, size_phys, freq)

    sim.run(warmup)
    queue = getattr(sim, "queue", None)
    if queue is not None:
        queue.finish()

    start_time = time.perf_counter()
    sim.run(steps)
    if queue is not None:
        queue.finish()
    return time.perf_counter() - start_time


def run_benchmark(shape=(100, 100, 100), steps=50, warmup=5, npml=10):
    Nx, Ny, Nz = shape
    total_cells = Nx * Ny * Nz
    dl = 1e-3
    freq = 6e9
    z_src = shape[2] - npml - 2
    source_bounds = (npml, Nx - npml, npml, Ny - npml)

    print("=" * 60)
    print(f"FDTD Performance Benchmark ({Nx}x{Ny}x{Nz} = {total_cells / 1e6:.2f}M cells)")
    print("=" * 60)

    ctr_phys = (Nx * dl / 2, Ny * dl / 2, Nz * dl / 2)
    size_phys = (Nx * dl * 0.6, Ny * dl * 0.6, Nz * dl * 0.6)

    print(f"Source: Jx sheet at z={z_src}, bounded to the non-PML XY region")
    print(f"Grid size: {total_cells / 1e6:.2f}M cells, {steps} steps after {warmup}-step warm-up")
    print("Monitor-inclusive timing includes each backend's near-to-far implementation.")

    workloads = (
        ("Field updates only", None),
        ("Field updates + near-to-far monitor", NumPyNear2FarMonitor),
    )
    for label, numpy_monitor_cls in workloads:
        opencl_monitor_cls = OpenCLNear2FarMonitor if numpy_monitor_cls is not None else None
        print(f"\n--- {label} ---")
        try:
            numpy_duration = _benchmark_backend(
                NumPyFDTD,
                numpy_monitor_cls,
                shape,
                dl,
                npml,
                z_src,
                source_bounds,
                freq,
                ctr_phys,
                size_phys,
                steps,
                warmup,
            )
            opencl_duration = _benchmark_backend(
                OpenCLFDTD,
                opencl_monitor_cls,
                shape,
                dl,
                npml,
                z_src,
                source_bounds,
                freq,
                ctr_phys,
                size_phys,
                steps,
                warmup,
            )
        except MemoryError as exc:
            raise SystemExit(f"ERROR: insufficient device memory — {exc}") from exc

        numpy_mcups = (total_cells * steps) / numpy_duration / 1e6
        opencl_mcups = (total_cells * steps) / opencl_duration / 1e6
        speedup = numpy_duration / opencl_duration
        print(f"NumPy:  {numpy_duration:.4f}s ({numpy_mcups:.2f} MCUPS)")
        print(f"OpenCL: {opencl_duration:.4f}s ({opencl_mcups:.2f} MCUPS)")
        print(f"OpenCL speedup: {speedup:.2f}x")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_benchmark()
