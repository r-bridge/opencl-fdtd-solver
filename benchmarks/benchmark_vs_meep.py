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

"""Comparative OpenCL-GPU vs MEEP-CPU benchmark.

Times sustained field updates only (sheet Ex source, no monitors). Reports the
median of several timed windows after an explicit warm-up so compile and
first-touch cost are excluded. Aborts before allocation if the model cannot
fit in device memory with headroom.
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np
from opencl_fdtd_solver import OpenCLFDTD

# Default stays within ~16 GB GPU budget after runtime headroom.
DEFAULT_SHAPE = (600, 600, 600)
DEFAULT_STEPS = 100
DEFAULT_WARMUP = 10
DEFAULT_REPEATS = 3
DEFAULT_NPML = 25


def estimate_gpu_memory_gb(shape, npml=DEFAULT_NPML):
    return OpenCLFDTD.estimate_device_memory_bytes(shape, npml) / (1024**3)


def parse_shape(text: str) -> tuple[int, int, int]:
    parts = [int(p) for p in text.lower().replace("x", ",").split(",")]
    if len(parts) == 1:
        return parts[0], parts[0], parts[0]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be N or Nx,Ny,Nz")
    return parts[0], parts[1], parts[2]


def run_opencl_benchmark(
    shape=DEFAULT_SHAPE,
    steps=DEFAULT_STEPS,
    warmup=DEFAULT_WARMUP,
    repeats=DEFAULT_REPEATS,
    npml=DEFAULT_NPML,
):
    """Runs the OpenCL solver on a GPU and returns sustained MCUPS stats."""
    print(f"Starting OpenCL FDTD Simulation ({shape[0]}x{shape[1]}x{shape[2]} grid)...")
    dl = 1e-3  # 1 mm
    freq = 6e9  # 6 GHz
    fwidth = 0.2 * freq

    os.environ["PYOPENCL_CTX"] = os.environ.get("PYOPENCL_CTX", "0")
    try:
        fdtd = OpenCLFDTD(shape, dl, npml=npml)
    except MemoryError as exc:
        raise SystemExit(f"ERROR: insufficient device memory — {exc}") from exc

    # Require a discrete/accelerator GPU — CPU OpenCL is not a meaningful
    # comparison against multi-threaded MEEP.
    import pyopencl as cl

    if not (fdtd.device.type & cl.device_type.GPU):
        raise RuntimeError(
            f"OpenCL device is not a GPU ({fdtd.device.name}). "
            "Set PYOPENCL_CTX to an NVIDIA/AMD GPU platform before benchmarking."
        )

    needed = estimate_gpu_memory_gb(shape, npml)
    budget = OpenCLFDTD.device_memory_budget_bytes(fdtd.device) / (1024**3)
    total = fdtd.device.global_mem_size / (1024**3)
    print(f"Device:  {fdtd.device.name}")
    print(f"VRAM:    {total:.2f} GB total, {budget:.2f} GB usable budget")
    print(f"Model:   ~{needed:.2f} GB float32 fields + face-local CPML buffers")

    z_src = shape[2] - npml - 2
    t0 = 5.0 / (np.pi * fwidth)
    sigma = 1.0 / (np.pi * fwidth)

    def source_cb(f):
        amp = np.exp(-0.5 * ((f.t - t0) / sigma) ** 2) * np.sin(2 * np.pi * freq * f.t)
        f.add_source_Ex(z_src, amp)

    fdtd._sources.append(source_cb)

    cells = shape[0] * shape[1] * shape[2]
    fdtd.run(warmup)
    fdtd.queue.finish()

    times = []
    rates = []
    for rep in range(repeats):
        start = time.perf_counter()
        fdtd.run(steps)
        fdtd.queue.finish()
        duration = time.perf_counter() - start
        mcups = (cells * steps) / duration / 1e6
        times.append(duration)
        rates.append(mcups)
        print(f"  OpenCL window {rep + 1}/{repeats}: {duration:.3f}s ({mcups:.1f} MCUPS)")

    return {
        "device": fdtd.device.name,
        "mem_gb": needed,
        "budget_gb": budget,
        "total_gb": total,
        "times": times,
        "mcups": rates,
        "time_s": statistics.median(times),
        "mcups_median": statistics.median(rates),
        "mcups_min": min(rates),
        "mcups_max": max(rates),
    }


def generate_meep_bench_script(shape=DEFAULT_SHAPE, steps=DEFAULT_STEPS, npml=DEFAULT_NPML):
    """Generates a MEEP script matched to the OpenCL benchmark geometry."""
    nx, ny, nz = shape
    # Match OpenCL Courant dt = 0.99 * dl / (c * sqrt(3)) in Meep units (c=1).
    until = steps * 0.99 / np.sqrt(3.0)
    z_src = nz - npml - 2
    z_center = z_src - nz / 2.0
    return f"""
import meep as mp
import time
import numpy as np

resolution = 1.0  # 1 voxel/mm
cell = mp.Vector3({nx}, {ny}, {nz})
boundary_layers = [mp.PML(thickness={float(npml)})]

f = 1.0 / 50.0  # 6 GHz in mm units
fwidth = 0.2 * f
t0 = 5.0 / (np.pi * fwidth)
sigma = 1.0 / (np.pi * fwidth)

def my_src_func(t):
    t_sec = t * 1e-3 / 299792458.0
    return np.exp(-0.5 * ((t_sec - t0) / sigma)**2) * np.sin(2 * np.pi * 6e9 * t_sec)

sources = [
    mp.Source(
        mp.CustomSource(src_func=my_src_func),
        component=mp.Ex,
        center=mp.Vector3(0, 0, {z_center}),
        size=mp.Vector3({nx}, {ny}, 0)
    )
]

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell,
    boundary_layers=boundary_layers,
    sources=sources,
    eps_averaging=False
)

# Warm-up a short advance so JIT / setup stay outside the timed window.
sim.run(until={0.99 / np.sqrt(3.0)})
start_time = time.perf_counter()
sim.run(until={until})
duration = time.perf_counter() - start_time
print("MEEP_BENCHMARK_DURATION:", duration)
"""


def run_meep_benchmark_in_docker(temp_dir, shape=DEFAULT_SHAPE, steps=DEFAULT_STEPS):
    """Executes the MEEP benchmark inside the local-pymeep Docker container."""
    if not shutil.which("docker"):
        print("ERROR: Docker not found. Cannot run MEEP benchmark.")
        return None

    script_path = os.path.join(temp_dir, "meep_bench.py")
    with open(script_path, "w") as f:
        f.write(generate_meep_bench_script(shape, steps))

    use_sg = sys.platform.startswith("linux")
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{temp_dir}:/work",
        "-w",
        "/work",
        "local-pymeep:latest",
        "python",
        "meep_bench.py",
    ]

    print("Starting MEEP Simulation inside Docker (CPU)...")
    try:
        if use_sg:
            cmd_str = " ".join(cmd)
            res = subprocess.run(
                ["sg", "docker", "-c", cmd_str],
                capture_output=True,
                text=True,
                check=True,
            )
        else:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)

        for line in res.stdout.splitlines():
            if "MEEP_BENCHMARK_DURATION:" in line:
                return float(line.split()[1])
        print("ERROR: MEEP did not report MEEP_BENCHMARK_DURATION.")
        print("MEEP stdout:\n", res.stdout)
        print("MEEP stderr:\n", res.stderr)
    except subprocess.CalledProcessError as e:
        print(f"ERROR running MEEP benchmark in Docker: {e}")
        print("stdout:\n", e.stdout)
        print("stderr:\n", e.stderr)
    except Exception as e:
        print(f"ERROR running MEEP benchmark in Docker: {e}")
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        type=parse_shape,
        default=DEFAULT_SHAPE,
        help="Grid size N or Nx,Ny,Nz (default 600)",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--npml", type=int, default=DEFAULT_NPML)
    parser.add_argument(
        "--skip-meep", action="store_true", help="Only time OpenCL (useful for large grids)"
    )
    args = parser.parse_args(argv)

    shape = args.shape
    steps = args.steps
    total_cells = shape[0] * shape[1] * shape[2]
    mem_gb = estimate_gpu_memory_gb(shape, args.npml)

    print("=" * 60)
    print("OPENCL GPU vs MEEP CPU COMPARATIVE BENCHMARK")
    print(f"Grid size: {shape[0]}x{shape[1]}x{shape[2]} = {total_cells / 1e6:.2f}M cells")
    print(f"Est. GPU model memory: {mem_gb:.2f} GB")
    print(f"Steps/window: {steps}  warmup: {args.warmup}  repeats: {args.repeats}")
    print("=" * 60 + "\n")

    cl = run_opencl_benchmark(
        shape, steps, warmup=args.warmup, repeats=args.repeats, npml=args.npml
    )
    print(
        f"[OK] OpenCL sustained median: {cl['time_s']:.2f}s "
        f"({cl['mcups_median']:.1f} MCUPS; "
        f"range {cl['mcups_min']:.1f}–{cl['mcups_max']:.1f})\n"
    )

    meep_time = None
    meep_mcups = None
    speedup = None
    if not args.skip_meep:
        with tempfile.TemporaryDirectory() as temp_dir:
            meep_time = run_meep_benchmark_in_docker(temp_dir, shape, steps)

        if meep_time is None:
            print("ERROR: Could not run MEEP benchmark. Aborting comparative results.")
            sys.exit(1)

        meep_mcups = (total_cells * steps) / meep_time / 1e6
        speedup = meep_time / cl["time_s"]
        print(f"[OK] MEEP Simulation completed: {meep_time:.2f}s ({meep_mcups:.2f} MCUPS)\n")

    print("=" * 60)
    print("COMPARISON SUMMARY:")
    print("=" * 60)
    print(f"Device (OpenCL):        {cl['device']}")
    print(
        f"Model memory:           {cl['mem_gb']:.2f} GB "
        f"(budget {cl['budget_gb']:.2f}/{cl['total_gb']:.2f} GB)"
    )
    if meep_time is not None:
        print(f"MEEP CPU (Docker):      {meep_time:7.2f}s ({meep_mcups:7.2f} MCUPS)")
    print(f"OpenCL FDTD (GPU):      {cl['time_s']:7.2f}s ({cl['mcups_median']:7.1f} MCUPS median)")
    if speedup is not None:
        print(f"Performance Ratio:      {speedup:.2f}x (MEEP time / OpenCL time)")
        if speedup > 1.0:
            print(f"OpenCL GPU is {speedup:.2f}x faster than MEEP on CPU.")
        else:
            print(f"MEEP CPU is {1 / speedup:.2f}x faster than OpenCL GPU.")
    print("=" * 60 + "\n")

    results_path = os.path.join(os.path.dirname(__file__), "last_meep_benchmark.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"device={cl['device']}\n")
        f.write(f"shape={shape[0]}x{shape[1]}x{shape[2]}\n")
        f.write(f"cells_m={total_cells / 1e6:.2f}\n")
        f.write(f"mem_gb={cl['mem_gb']:.2f}\n")
        f.write(f"budget_gb={cl['budget_gb']:.2f}\n")
        f.write(f"steps={steps}\n")
        f.write(f"warmup={args.warmup}\n")
        f.write(f"repeats={args.repeats}\n")
        f.write(f"opencl_s_median={cl['time_s']:.4f}\n")
        f.write(f"opencl_mcups_median={cl['mcups_median']:.4f}\n")
        f.write(f"opencl_mcups_min={cl['mcups_min']:.4f}\n")
        f.write(f"opencl_mcups_max={cl['mcups_max']:.4f}\n")
        if meep_time is not None:
            f.write(f"meep_s={meep_time:.4f}\n")
            f.write(f"meep_mcups={meep_mcups:.4f}\n")
            f.write(f"speedup={speedup:.4f}\n")
    print(f"Wrote results to {results_path}")


if __name__ == "__main__":
    main()
