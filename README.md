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
*   **Cell-wise materials:** Nondispersive scalar εᵣ via `set_epsilon`. Subpixel averaging / geometry meshing is left to the caller (see §6).
*   **Dependency-Free:** Pure Python package with minimal requirements (no C compiler or toolchains required at install time).

---

## 3. Installation
Ensure you have an OpenCL platform (NVIDIA CUDA, AMD, Intel, or POCL) installed, then:

```bash
pip install -e .
# Optional: Meep baseline / coverage tooling
pip install -e ".[test]"
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

### Lint / format / typecheck
CI also runs Ruff (lint + format) and Mypy on `opencl_fdtd_solver/`:

```bash
pip install -e ".[lint]"
ruff check opencl_fdtd_solver tests benchmarks setup.py
ruff format --check opencl_fdtd_solver tests benchmarks setup.py
mypy opencl_fdtd_solver
```

Optional local hooks: `pip install pre-commit && pre-commit install`.

### NumPy CPML storage
- Default NumPy reference (`NumPyFDTD`) uses volume CPML with an option to reduce memory via `psi_dtype`.
- An experimental `NumPyFDTD_FaceCPML` scaffolding mirrors the OpenCL face-local CPML layout and will be enabled after validation.

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

### Mid-plane Ex golden images + discrepancy report
CI regenerates generic side-by-side mid-plane Ex triptychs (`OpenCL | Meep | residual`) and an objective **discrepancy report**, and requires both to match committed baselines under `tests/meep_validation/baselines/`:

- per-case PNGs / float planes
- [`DISCREPANCY_REPORT.md`](tests/meep_validation/baselines/DISCREPANCY_REPORT.md) (human-readable metrics for repository viewers)
- `discrepancy_report.json` (exact JSON enforced by CI)

Cases are abstract (`vacuum_sheet`, `dielectric_block`) — matched Courant, SI `Jx` sheet with Meep-matched rim taper (`Ex += −dt/(ε₀εᵣ)J`, trimmed out of PML), CPML. Quality gates also fail if mean Pearson correlation, LMS scale, or residual energy worsen beyond fixed floors.

```bash
# Refresh all committed images, planes, and reports (use POCL/CPU so CI matches):
IGNORE_GPU=NVIDIA,AMD PYOPENCL_CTX=0 python -m tests.meep_validation.update_plane_baselines

# Enforce parity (same check as CI):
PYOPENCL_CTX=0 python -m unittest tests.test_meep_plane_baselines -v
```

### Near-to-far golden patterns + discrepancy report
CI also gates committed XZ far-field `|S|(θ)` overlays and EH probes from `OpenCLNear2FarMonitor` vs Meep (`vacuum_farfield`, `dielectric_sphere_farfield`) under the same `baselines/` tree:

- per-case `pattern_xz.png` / `ocl_S_db.npy` / `meep_S_db.npy` / EH arrays
- [`DISCREPANCY_REPORT_FARFIELD.md`](tests/meep_validation/baselines/DISCREPANCY_REPORT_FARFIELD.md)
- `discrepancy_report_farfield.json`

```bash
IGNORE_GPU=NVIDIA,AMD PYOPENCL_CTX=0 python -m tests.meep_validation.update_farfield_baselines
PYOPENCL_CTX=0 python -m unittest tests.test_meep_farfield_baselines -v
```

OpenCL ↔ NumPy field/monitor parity remains in `tests/test_solver.py` (always run in CI).

---

## 6. Accuracy for Practical Applications

This package is a **2nd-order Yee + CPML kernel**. For nondispersive scalar-ε problems on a matched uniform grid (same Δx, Courant, and cell-wise materials), it is in the **same accuracy class as MEEP’s default FDTD**. MEEP is not inherently more accurate for that shared physics; the repo’s value proposition is GPU throughput, not higher-order fidelity.

**Evidence (MEEP-relative, abstract cases):** mid-plane Ex shape agrees at ~98% Pearson correlation with ~3% aligned residual energy; far-field main-lobe |S|(θ) differs by ~0.7 dB (vacuum) to ~2.8 dB (εᵣ=4 sphere) under the CI masks. Those baselines use hard voxel materials (`eps_averaging=False` on the MEEP side) and do not certify a specific device design.

**When MEEP (or another full-featured solver) is usually ahead in practice:** dispersive / lossy / magnetic media, PEC/periodic/symmetry BCs, double precision, or any workflow that relies on a built-in geometry stack you do not provide yourself.

**Subpixel averaging is intentionally out of scope.** The solver only accepts a cell-wise scalar εᵣ via `set_epsilon`; geometry sampling and effective-medium construction belong in the application layer. Typical choices include ~4³ subvoxels per Yee cell. At least two approaches fit this design:

1. **Object definitions as OpenCL kernels**, with subvoxel sampling / averaging also performed in OpenCL.
2. **GPU-accelerated STL tiled rendering** with averaging into the Yee ε grid.

Supply an adequately resolved, averaged ε array (and match sources / Courant carefully), and expect accuracy comparable to MEEP on the same grid for supported physics. Remaining error is dominated by mesh density, float32 dynamic range, and PML / source / near-to-far details—not by the absence of in-solver averaging.

---

## 7. Performance Benchmarks
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
| Relative (matched) | 400³ × 100 steps | ~7970 MCUPS | ~82 MCUPS | **~97×** |
| Sustained OpenCL | 600³ × 100 steps | ~8630 MCUPS | — | — |
| Near-capacity OpenCL | 750³ × 80 steps | ~9430 MCUPS | — | — |

Near-to-far uses a **single fused tangential face-DFT kernel** per step (not one launch per field×face). On a 200³ smoke test with a large Huygens box, sustained rates stay near field-only (~7.3k vs ~7.4k MCUPS). Absolute MCUPS still depends on free VRAM — trust local runs over the table.
