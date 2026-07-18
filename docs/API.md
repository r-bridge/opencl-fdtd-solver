# API reference

Public surface of `opencl_fdtd_solver` (what `__all__` exports). Physics background: [`PHYSICS.md`](PHYSICS.md).

```python
from opencl_fdtd_solver import (
    C0, MU0, EPS0, ETA0,
    OpenCLFDTD,
    NumPyFDTD,
    NumPyFDTD_FaceCPML,
    OpenCLNear2FarMonitor,
    NumPyNear2FarMonitor,
    StepCallback,
)
```

---

## Constants

| Name | Meaning | SI unit |
|---|---|---|
| `C0` | Speed of light | m/s |
| `MU0` | Vacuum permeability | H/m |
| `EPS0` | Vacuum permittivity | F/m |
| `ETA0` | Vacuum impedance √(μ₀/ε₀) | Ω |

---

## Solvers

Both OpenCL and NumPy engines share the same public control flow: set materials → register sources/monitors → `step` / `run`. Sources fire after the H update (at half-step time); monitors fire after the full Yee step.

### Shared registration (`SourceMonitorMixin`)

| Method | Role |
|---|---|
| `add_source(cb)` | Register `cb(fdtd)` after each H update |
| `add_monitor(cb)` | Register `cb(fdtd)` after each full step |
| `clear_sources()` | Drop all source callbacks |
| `clear_monitors()` | Drop all monitor callbacks |

`StepCallback` is the typing `Protocol` for those callables.

### `OpenCLFDTD` (production)

```python
OpenCLFDTD(shape, dl, npml=20, dtype=np.float32, ctx=None, queue=None)
```

| Arg | Description |
|---|---|
| `shape` | `(Nx, Ny, Nz)` Yee cells |
| `dl` | Uniform cell size (m) |
| `npml` | CFS-CPML thickness (cells); `0` disables PML |
| `dtype` | `np.float32` (default) or `np.float64` |
| `ctx`, `queue` | Optional existing PyOpenCL context/queue |

**FP64** rebuilds the same kernels with `real=double`. Requires device `cl_khr_fp64` (or `cl_amd_fp64`) and `cl_khr_int64_base_atomics` (near-to-far). Expect ~2× wall time on bandwidth-bound grids. Missing extensions → `ValueError`. Oversized models → `MemoryError` before allocation.

| Attribute / property | Notes |
|---|---|
| `Nx, Ny, Nz`, `dl`, `npml`, `dt`, `t`, `step_num` | Grid and time state |
| `dtype`, `real`, `complex_dtype` | Computation dtypes |
| `device`, `ctx`, `queue` | OpenCL runtime |
| `Ex`…`Hz` | Host copies of device fields (`dtype`) |

| Method | Notes |
|---|---|
| `set_epsilon(eps)` | Cell-wise scalar εᵣ, shape `(Nx,Ny,Nz)`; builds Yee-edge `ce` coeffs |
| `add_source_Jx(z, Jx, i0=…, i1=…, j0=…, j1=…, *, rim_taper=False, rim_edge=0.8, rim_renorm=True)` | SI Jx (A/m²) on a constant-z Ex sheet: `Ex += -dt/(ε₀ εᵣ) Jx` |
| `add_source_Ex(…)` | **Deprecated** soft Ex add; prefer `add_source_Jx` |
| `step()` | One Yee step (H → sources → E → monitors) |
| `run(n_steps, progress_every=0)` | Loop `step` |
| `read_point(name, i, j, k)` | Single cell without full-field download (`name` in `Ex`…`Hz`) |
| `estimate_device_memory_bytes(shape, npml, dtype=…)` | Static GPU memory estimate |

Typical drive loop:

```python
import numpy as np
from opencl_fdtd_solver import OpenCLFDTD

sim = OpenCLFDTD((128, 128, 128), dl=1e-3, npml=12)  # or dtype=np.float64
sim.set_epsilon(np.ones(sim.Nx * sim.Ny * sim.Nz).reshape(sim.Nx, sim.Ny, sim.Nz))

def src(f):
    f.add_source_Jx(f.Nz // 2, np.sin(2 * np.pi * 5e9 * f.t), rim_taper=True)

sim.add_source(src)
sim.run(200)
```

### `NumPyFDTD` (CPU reference)

```python
NumPyFDTD(shape, dl, npml=20, dtype=np.float32, psi_dtype=None)
```

Same Yee/CPML physics and source/monitor API as OpenCL. CPML ψ arrays are **full volume** (`Nx×Ny×Nz` × 12). Optional `psi_dtype` (e.g. `float16`) reduces ψ memory at the cost of parity.

### `NumPyFDTD_FaceCPML`

Subclass of `NumPyFDTD` with OpenCL-matched **face-striped** ψ storage (`CPML_STORAGE = "face"`). Prefer for large CPU grids. Same public API; numerically equivalent when `c≡0` outside the PML.

---

## Near-to-far monitors

Huygens box DFT at one frequency, then far E/H from equivalent surface currents.

```python
OpenCLNear2FarMonitor(fdtd, center, size, freq)
NumPyNear2FarMonitor(fdtd, center, size, freq)
```

| Arg | Description |
|---|---|
| `fdtd` | Solver instance (OpenCL monitor requires `OpenCLFDTD`) |
| `center` | Box center `(x,y,z)` in metres |
| `size` | Box extents `(sx,sy,sz)` in metres |
| `freq` | DFT frequency (Hz) |

Construction **registers** the monitor on `fdtd` automatically.

| Method | Notes |
|---|---|
| `get_farfield(obs)` | Complex `(Ex,Ey,Ez,Hx,Hy,Hz)` at one point (m) |
| `get_farfields(points)` | Shape `(n,6)` for many points (`OpenCLNear2FarMonitor`) |
| `farfield_polar_xz(*, distance_m=1000, n_angles=73)` | XZ cut: angles (deg), \|S\| (dB) |
| `snapshot_dft()` / `dft_relative_change()` | Device-side convergence helper (OpenCL) |
| `fetch_dft_fields()` | Download face DFT into sparse host volumes (debug) |

OpenCL path accumulates **face-packed** DFT on device (dtype matches `fdtd.dtype`). NumPy path keeps host volume DFTs.

```python
from opencl_fdtd_solver import OpenCLFDTD, OpenCLNear2FarMonitor

sim = OpenCLFDTD((64, 64, 64), 2e-3, npml=8)
ctr = (64 * 2e-3 / 2,) * 3
mon = OpenCLNear2FarMonitor(sim, ctr, (20e-3, 20e-3, 20e-3), freq=5e9)
sim.add_source(lambda f: f.add_source_Jx(20, 1.0))
sim.run(300)
angles, db = mon.farfield_polar_xz(distance_m=1.0, n_angles=73)
eh = mon.get_farfield((0.0, 0.0, 1.0))
```

---

## Conventions

- **Units:** SI throughout (see [`PHYSICS.md`](PHYSICS.md) §2).
- **Indexing:** Fields are `(i, j, k)` with `k` the fastest axis on device.
- **Time:** `dt = 0.99 · dl / (c√3)`; `t` advances by `dt` each `step`.
- **Materials:** Scalar nondispersive εᵣ only; μ=μ₀. No PEC/periodic BCs in-core.
- **Stability / memory:** Prefer checking `estimate_device_memory_bytes` on large grids; the OpenCL constructor already enforces headroom.

Internal modules (`cpml`, `materials`, `kernels`, …) are implementation details and may change without notice.
