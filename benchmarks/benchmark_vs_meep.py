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

"""Comparative OpenCL-GPU vs MEEP-CPU benchmark near GPU VRAM capacity.

Default grid is 560^3 (~176M Yee cells, ~12.4 GB of float32 field/CPML
buffers), sized to sit near the limit of a 16 GB GPU without spilling into
host memory (which collapses MCUPS).
"""

import os
import sys
import time
import tempfile
import subprocess
import shutil
import numpy as np
from opencl_fdtd_solver import OpenCLFDTD

# 19 float32 volumes: Ex/Ey/Ez/Hx/Hy/Hz/eps + 12 CPML psi arrays
BYTES_PER_CELL = 19 * 4

# Near-limit default for a 16 GB discrete GPU (validated on RTX 5080 16 GB).
DEFAULT_SHAPE = (560, 560, 560)
DEFAULT_STEPS = 200


def estimate_gpu_memory_gb(shape):
    return (shape[0] * shape[1] * shape[2] * BYTES_PER_CELL) / (1024 ** 3)


def run_opencl_benchmark(shape=DEFAULT_SHAPE, steps=DEFAULT_STEPS):
    """Runs the OpenCL solver on the first available GPU device."""
    print(f"Starting OpenCL FDTD Simulation ({shape[0]}x{shape[1]}x{shape[2]} grid)...")
    dl = 1e-3  # 1 mm
    npml = 25
    freq = 6e9  # 6 GHz
    fwidth = 0.2 * freq

    os.environ["PYOPENCL_CTX"] = os.environ.get("PYOPENCL_CTX", "0")
    fdtd = OpenCLFDTD(shape, dl, npml=npml)

    # Require a discrete/accelerator GPU — CPU OpenCL is not a meaningful
    # comparison against multi-threaded MEEP.
    dev_type = fdtd.device.type
    import pyopencl as cl
    if not (dev_type & cl.device_type.GPU):
        raise RuntimeError(
            f"OpenCL device is not a GPU ({fdtd.device.name}). "
            "Set PYOPENCL_CTX to an NVIDIA/AMD GPU platform before benchmarking."
        )

    mem_gb = estimate_gpu_memory_gb(shape)
    print(f"Device:  {fdtd.device.name}")
    print(f"VRAM:    {fdtd.device.global_mem_size / (1024 ** 3):.2f} GB reported")
    print(f"Model:   ~{mem_gb:.2f} GB of float32 field/CPML buffers")

    z_src = shape[2] - npml - 2
    t0 = 5.0 / (np.pi * fwidth)
    sigma = 1.0 / (np.pi * fwidth)

    def source_cb(f):
        amp = np.exp(-0.5 * ((f.t - t0) / sigma) ** 2) * np.sin(2 * np.pi * freq * f.t)
        f.add_source_Ex(z_src, amp)

    fdtd._sources.append(source_cb)

    # Warm-up so compile / first-touch are outside the timed window.
    fdtd.run(2)
    fdtd.queue.finish()

    start_time = time.perf_counter()
    fdtd.run(steps)
    fdtd.queue.finish()
    duration = time.perf_counter() - start_time
    mcups = (shape[0] * shape[1] * shape[2] * steps) / duration / 1e6
    return duration, mcups, fdtd.device.name, mem_gb


def generate_meep_bench_script(shape=DEFAULT_SHAPE, steps=DEFAULT_STEPS, npml=25):
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

start_time = time.perf_counter()
sim.run(until={until})
duration = time.perf_counter() - start_time
print("MEEP_BENCHMARK_DURATION:", duration)
"""


def run_meep_benchmark_in_docker(temp_dir, shape=DEFAULT_SHAPE, steps=DEFAULT_STEPS):
    """Executes the MEEP benchmark inside the local-pymeep Docker container."""
    if not shutil.which("docker"):
        print("Docker not found. Cannot run MEEP benchmark.")
        return None

    script_path = os.path.join(temp_dir, "meep_bench.py")
    with open(script_path, "w") as f:
        f.write(generate_meep_bench_script(shape, steps))

    use_sg = sys.platform.startswith("linux")
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{temp_dir}:/work",
        "-w", "/work",
        "local-pymeep:latest",
        "python", "meep_bench.py",
    ]

    print("Starting MEEP Simulation inside Docker (CPU)...")
    try:
        if use_sg:
            cmd_str = " ".join(cmd)
            res = subprocess.run(
                ["sg", "docker", "-c", cmd_str],
                capture_output=True, text=True, check=True,
            )
        else:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)

        for line in res.stdout.splitlines():
            if "MEEP_BENCHMARK_DURATION:" in line:
                return float(line.split()[1])
        print("MEEP stdout:\n", res.stdout)
        print("MEEP stderr:\n", res.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running MEEP benchmark in Docker: {e}")
        print("stdout:\n", e.stdout)
        print("stderr:\n", e.stderr)
    except Exception as e:
        print(f"Error running MEEP benchmark in Docker: {e}")
    return None


def main():
    shape = DEFAULT_SHAPE
    steps = DEFAULT_STEPS
    total_cells = shape[0] * shape[1] * shape[2]
    mem_gb = estimate_gpu_memory_gb(shape)

    print("=" * 60)
    print("OPENCL GPU vs MEEP CPU COMPARATIVE BENCHMARK")
    print(f"Grid size: {shape[0]}x{shape[1]}x{shape[2]} = {total_cells / 1e6:.2f}M cells")
    print(f"Est. GPU model memory: {mem_gb:.2f} GB (near 16 GB VRAM capacity)")
    print(f"Steps:     {steps}")
    print("=" * 60 + "\n")

    cl_time, cl_mcups, device_name, _ = run_opencl_benchmark(shape, steps)
    print(f"[OK] OpenCL FDTD completed: {cl_time:.2f}s ({cl_mcups:.2f} MCUPS)\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        meep_time = run_meep_benchmark_in_docker(temp_dir, shape, steps)

    if meep_time is None:
        print("Could not run MEEP benchmark. Aborting comparative results.")
        sys.exit(1)

    meep_mcups = (total_cells * steps) / meep_time / 1e6
    print(f"[OK] MEEP Simulation completed: {meep_time:.2f}s ({meep_mcups:.2f} MCUPS)\n")

    print("=" * 60)
    print("COMPARISON SUMMARY:")
    print("=" * 60)
    print(f"Device (OpenCL):        {device_name}")
    print(f"Model memory:           {mem_gb:.2f} GB / ~16 GB GPU")
    print(f"MEEP CPU (Docker):      {meep_time:7.2f}s ({meep_mcups:6.2f} MCUPS)")
    print(f"OpenCL FDTD (GPU):      {cl_time:7.2f}s ({cl_mcups:6.2f} MCUPS)")

    speedup = meep_time / cl_time
    print(f"Performance Ratio:      {speedup:.2f}x")
    if speedup > 1.0:
        print(f"OpenCL GPU is {speedup:.2f}x faster than MEEP on CPU.")
    else:
        print(f"MEEP CPU is {1 / speedup:.2f}x faster than OpenCL GPU.")
    print("=" * 60 + "\n")

    results_path = os.path.join(os.path.dirname(__file__), "last_meep_benchmark.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"device={device_name}\n")
        f.write(f"shape={shape[0]}x{shape[1]}x{shape[2]}\n")
        f.write(f"cells_m={total_cells / 1e6:.2f}\n")
        f.write(f"mem_gb={mem_gb:.2f}\n")
        f.write(f"steps={steps}\n")
        f.write(f"opencl_s={cl_time:.4f}\n")
        f.write(f"opencl_mcups={cl_mcups:.4f}\n")
        f.write(f"meep_s={meep_time:.4f}\n")
        f.write(f"meep_mcups={meep_mcups:.4f}\n")
        f.write(f"speedup={speedup:.4f}\n")
    print(f"Wrote results to {results_path}")


if __name__ == "__main__":
    main()
