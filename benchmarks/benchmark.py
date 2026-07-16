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

import os
import time

import numpy as np
from opencl_fdtd_solver import NumPyFDTD, NumPyNear2FarMonitor, OpenCLFDTD, OpenCLNear2FarMonitor


def run_benchmark(shape=(100, 100, 100), steps=50, warmup=5):
    Nx, Ny, Nz = shape
    total_cells = Nx * Ny * Nz
    dl = 1e-3
    npml = 10
    freq = 6e9

    print("=" * 60)
    print(f"FDTD Performance Benchmark ({Nx}x{Ny}x{Nz} = {total_cells / 1e6:.2f}M cells)")
    print("=" * 60)

    # 1. Benchmark NumPy CPU reference
    print("\n--- [1/2] Benchmarking NumPy CPU Reference ---")
    np_sim = NumPyFDTD(shape, dl, npml=npml)
    z_src = shape[2] - npml - 2
    np_sim.add_source(lambda f: setattr(f, "Ex", f.Ex + np.sin(2 * np.pi * freq * f.t)))

    ctr_phys = (Nx * dl / 2, Ny * dl / 2, Nz * dl / 2)
    size_phys = (Nx * dl * 0.6, Ny * dl * 0.6, Nz * dl * 0.6)
    NumPyNear2FarMonitor(np_sim, ctr_phys, size_phys, freq)

    np_sim.run(warmup)
    start_time = time.perf_counter()
    np_sim.run(steps)
    np_duration = time.perf_counter() - start_time
    np_mcups = (total_cells * steps) / np_duration / 1e6
    print(f"NumPy duration: {np_duration:.4f} seconds ({np_mcups:.2f} MCUPS)")

    # 2. Benchmark OpenCL GPU/CPU
    print("\n--- [2/2] Benchmarking OpenCL Solver ---")
    os.environ["PYOPENCL_CTX"] = os.environ.get("PYOPENCL_CTX", "0")
    try:
        cl_sim = OpenCLFDTD(shape, dl, npml=npml)
    except MemoryError as exc:
        raise SystemExit(f"ERROR: insufficient device memory — {exc}") from exc
    cl_sim.add_source(lambda f: f.add_source_Ex(z_src, np.sin(2 * np.pi * freq * f.t)))
    OpenCLNear2FarMonitor(cl_sim, ctr_phys, size_phys, freq)

    cl_sim.run(warmup)
    cl_sim.queue.finish()
    start_time = time.perf_counter()
    cl_sim.run(steps)
    cl_sim.queue.finish()
    cl_duration = time.perf_counter() - start_time
    cl_mcups = (total_cells * steps) / cl_duration / 1e6
    print(f"OpenCL duration: {cl_duration:.4f} seconds ({cl_mcups:.2f} MCUPS)")

    print("\n" + "=" * 60)
    print("Benchmark Comparison Summary:")
    print("=" * 60)
    print(f"Grid size: {total_cells / 1e6:.2f}M cells, {steps} steps (after {warmup}-step warm-up)")
    print(f"NumPy CPU:  {np_duration:7.4f}s ({np_mcups:6.2f} MCUPS)")
    print(f"OpenCL:     {cl_duration:7.4f}s ({cl_mcups:6.2f} MCUPS)")
    speedup = np_duration / cl_duration
    print(f"Speedup:    {speedup:.2f}x faster using OpenCL")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_benchmark()
