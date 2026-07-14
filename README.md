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
A generic unit test suite checks OpenCL ↔ NumPy field parity, near-to-far monitors, memory budgeting, and harness helpers. Aim for **≥90%** line coverage of `opencl_fdtd_solver/` (enforced in CI):

```bash
PYOPENCL_CTX=0 python tests/run_coverage.py
# or step-by-step:
PYOPENCL_CTX=0 python -m coverage run -m unittest \
  tests.test_solver tests.test_unit_engine tests.test_unit_monitors tests.test_unit_harness -v
python -m coverage report --fail-under=90
```

Minimal smoke without coverage:

```bash
PYOPENCL_CTX=0 python -m unittest tests.test_solver tests.test_unit_engine tests.test_unit_monitors tests.test_unit_harness -v
```

---

## 5. MEEP Correctness Comparison
Supported features are validated extensively against [MEEP](https://meep.readthedocs.io/) (local install or `local-pymeep` Docker). The suite hard-fails if MEEP cannot run (set `ALLOW_SKIP_MEEP=1` only for environments without MEEP).

```bash
PYOPENCL_CTX=0 python -m unittest tests.test_meep_validation -v
# or
PYOPENCL_CTX=0 python tests/compare_with_meep.py
```

| Case | What it checks | Tolerance |
|---|---|---|
| Near-field Ex DFT | Yee + CPML + Ex sheet at interior probes | peak-normalized max err `< 0.20` |
| Far-field \|S\|(θ) vacuum | Near-to-far XZ pattern vs Meep | main lobe (`mask_db=-12`) `< 2.5 dB` |
| Far-field vector EH | Ex/Hy on +z; deep null on +x | pol. err `< 0.35`; \|E(+x)\|/\|E(+z)\| `< 0.05` |
| Dielectric sphere εᵣ=4 | Material + pattern | main lobe `< 3 dB` |
| PML energy decay | Late/peak Ex energy ratio | both `< 0.05`, ratios within 100× |

OpenCL ↔ NumPy field/monitor parity remains in `tests/test_solver.py` (always run in CI).

---

## 6. Performance Benchmarks
Two benchmarks are provided: a small NumPy vs OpenCL check, and an OpenCL GPU vs MEEP CPU comparison.

Throughput is **sustained** cell-updates after warm-up (`queue.finish()` around each timed window). Peak one-shot numbers are easy to overstate; if the model barely fits in VRAM the driver can page to host RAM and effective throughput can drop by ~10× without a clear OpenCL error. The solver therefore **raises `MemoryError` before allocation** when the estimate exceeds usable device memory (total minus 12% / 512 MiB headroom).

### Benchmark 1: NumPy CPU Reference vs OpenCL (1.0M Cells)
Includes near-to-far monitors (workload comparable to interactive use):

```bash
PYOPENCL_CTX=0 python benchmarks/benchmark.py
```
Re-run locally; do not treat older printed MCUPS as authoritative across machines or OpenCL backends.

### Benchmark 2: MEEP CPU vs OpenCL GPU
Compares MEEP (CPU, Docker `local-pymeep:latest`) against this solver on a GPU. Default grid is **600³** (~216M cells, ~6.4 GB) for stable VRAM headroom. Use `--shape 750` only if your GPU has clear free memory after the headroom check.

```bash
PYOPENCL_CTX=0 python -u benchmarks/benchmark_vs_meep.py
# OpenCL only (large grids / skip Docker):
PYOPENCL_CTX=0 python -u benchmarks/benchmark_vs_meep.py --shape 600 --skip-meep
```

Measured on NVIDIA GeForce RTX 5080 (15.92 GB reported), AMD Ryzen 9 7945HX, field updates + Ex sheet source, **no monitors** (median of 3 timed windows after warm-up):

| Case | Grid | OpenCL | MEEP CPU | Speedup |
|---|---:|---:|---:|---:|
| Relative (matched) | 400³ × 100 steps | ~8160 MCUPS | ~49 MCUPS | **~166×** |
| Sustained OpenCL | 600³ × 100 steps | ~8740 MCUPS | — | — |
| Near-capacity OpenCL | 750³ × 80 steps | ~8480 MCUPS | — | — |

Near-to-far uses a **single fused tangential face-DFT kernel** per step (not one launch per field×face). On a 200³ smoke test with a large Huygens box, sustained rates stay near field-only (~7.3k vs ~7.4k MCUPS). Absolute MCUPS still depends on free VRAM — trust local runs over the table.
