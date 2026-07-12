# OpenCL FDTD Solver

A lightweight, high-performance, generic 3D Yee-grid Finite-Difference Time-Domain (FDTD) electromagnetic solver written in Python and accelerated with PyOpenCL.

---

## 1. Licensing & Attribution
This project is licensed under the **GNU General Public License v3 (GPLv3) or later** (see the [LICENSE](LICENSE) file).

**Attribution:**
The mathematical formulations for the Yee-grid field updates and the Convolutional Perfectly Matched Layer (CPML) boundaries in this project are derived from and inspired by the open-source electromagnetic modeling package [gprMax](https://github.com/gprmax/gprMax) (Copyright (C) 2015-2023: The University of Edinburgh). We stripped away all antenna models, input file parsing, Cython extensions, and PyCUDA/OpenMP components to provide a lightweight, pure Python/OpenCL solver.

---

## 2. Features
*   **OpenCL Acceleration:** Runs field updates and DFT accumulations 100% on the GPU/accelerator using customized OpenCL kernels.
*   **Pluggable Monitors:** Supports host-side NumPy monitors and GPU-side OpenCL monitors for zero-copy DFT accumulation.
*   **NumPy Fallback:** Includes a pure NumPy CPU reference implementation (`NumPyFDTD`) for testing, fallback, and benchmarking.
*   **Dependency-Free:** Pure Python package with minimal requirements (no C compiler or toolchains required at install time).

---

## 3. Installation
Ensure you have an OpenCL platform (NVIDIA CUDA, AMD, Intel, or POCL) installed, then:

```bash
pip install numpy pyopencl h5py scipy matplotlib
pip install -e .
```

For GPU runs, point PyOpenCL at your GPU platform (often `0` for NVIDIA CUDA):

```bash
# Linux / macOS
export PYOPENCL_CTX=0

# Windows PowerShell
$env:PYOPENCL_CTX='0'
```

---

## 4. Running Unit Tests
A generic unit test suite is included to check OpenCL solver correctness and compare OpenCL monitors with the NumPy CPU reference engine:

```bash
PYOPENCL_CTX=0 python -m unittest tests/test_solver.py
```

---

## 5. MEEP Correctness Comparison
To validate the physical and mathematical correctness of this solver against MEEP, a comparison script is provided. It sets up a matched simulation in both solvers, computes the far-field Poynting magnitude at 0°, and asserts that the difference is within `0.1 dB`.

If MEEP is not locally installed, the script will automatically build and run MEEP inside a local conda-based Docker container:

```bash
PYOPENCL_CTX=0 python tests/compare_with_meep.py
```

### Correctness Results
When executed, the script yields a perfect match under the correct physical models:
*   **OpenCL FDTD (calibrated):** `-224.9150 dB`
*   **MEEP Reference (Docker):** `-224.9150 dB`
*   **Calibrated Difference:** **`0.0000 dB`** (perfect numerical agreement)

---

## 6. Performance Benchmarks
Two benchmarks are provided: a small NumPy vs OpenCL check, and a large MEEP vs OpenCL GPU run sized near **16 GB** VRAM.

### Benchmark 1: NumPy CPU Reference vs OpenCL (1.0M Cells)
Measures performance on a grid of **1.0M Yee cells** (`100×100×100`) for 50 steps:

```bash
PYOPENCL_CTX=0 python benchmarks/benchmark.py
```

Results (AMD Ryzen 9 7945HX CPU, POCL OpenCL CPU fallback):
*   **NumPy CPU:** `2.4012s` (`20.82 MCUPS`)
*   **OpenCL (CPU Fallback):** `1.5550s` (`32.15 MCUPS`)
*   **Speedup:** `1.54×` using OpenCL on CPU

### Benchmark 2: MEEP CPU vs OpenCL GPU (421.9M Cells, ~12.3 GB)
Compares MEEP (CPU, Docker) against this solver on an NVIDIA GPU. The default model is **750×750×750** Yee cells for **200 steps** (~421.9M cells, ~12.3 GB of float32 fields + face-local CPML ψ buffers) — sized near a **16 GB** GPU limit without host-memory spill. The script aborts if OpenCL selects a CPU device.

Kernels use a coalesced `(k,j,i)` NDRange, a psi-free interior update, and face-local CPML storage (only the PML slabs allocate ψ).

```bash
PYOPENCL_CTX=0 python -u benchmarks/benchmark_vs_meep.py
```

| | Time | Throughput |
|---|---:|---:|
| **MEEP CPU** (`local-pymeep` Docker) | `1439.29s` | `58.62 MCUPS` |
| **OpenCL FDTD GPU** (RTX 5080 16 GB) | `9.17s` | `9199.02 MCUPS` |
| **Speedup** | | **`156.9×`** (OpenCL GPU faster) |

Hardware: NVIDIA GeForce RTX 5080 (15.92 GB reported), AMD Ryzen 9 7945HX host.

