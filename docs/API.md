# API reference

Public surface of `opencl_fdtd_solver` (what `__all__` exports). Physics background: [`PHYSICS.md`](PHYSICS.md).

**Maintenance:** keep this file consistent with the code. Agents and contributors must update it whenever `__all__` or public methods change; `tests/test_api_docs.py` (and `AGENTS.md`) enforce that.

```python
from opencl_fdtd_solver import (
    C0,
    MU0,
    EPS0,
    ETA0,
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

During a source callback, `add_source_Jx` stages the current contribution until
after the E update: the old field decays, but the newly injected current does
not acquire an extra loss factor. The staged contribution is not visible to
field reads inside that callback. Outside source callbacks, `add_source_Jx`
applies immediately. Legacy `add_source_Ex` always applies immediately.

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
OpenCLFDTD(shape, dl, npml=20, dtype=np.float32, ctx=None, queue=None, accum_fp64=None)
```

| Arg | Description |
|---|---|
| `shape` | Three positive integers `(Nx, Ny, Nz)` |
| `dl` | Finite, positive uniform cell size (m) |
| `npml` | Nonnegative integer CFS-CPML thickness; `2*npml <= min(shape)`; `0` disables PML |
| `dtype` | `np.float32` (default) or `np.float64` |
| `ctx`, `queue` | Optional existing PyOpenCL context and queue. A supplied queue provides the context if omitted and selects the device used for kernel builds. If both are supplied, the queue must belong to `ctx` |
| `accum_fp64` | With FP32 fields, `None` automatically enables FP64 monitor DFT and near-to-far accumulation on queue devices with FP64 and `cl_khr_int64_base_atomics`. Use `True` to require FP64 or `False` to keep FP32 accumulation. FP64 fields always use FP64 accumulation |

**Accumulator precision is the device default.** Before the automatic
setting, FP32 fields always accumulated in FP32 unless `accum_fp64=True` was
passed. On a device with the two extensions the monitor DFT buffers
(`Ex_dft_buf` … `Hz_dft_buf`) now hold `complex128` rather than `complex64`.
Code that reads or writes those buffers directly must take the element type
from `fdtd.accreal_complex_dtype`: a `complex64` array of `n_face_samples`
elements fills only half of a `complex128` buffer and is read back as
different numbers, without an error. Pass `accum_fp64=False` to keep the
previous layout.

**FP64** rebuilds the same kernels with `real=double`. Requires device `cl_khr_fp64` (or `cl_amd_fp64`) and `cl_khr_int64_base_atomics` (near-to-far). Expect ~2× wall time on bandwidth-bound grids. Missing extensions → `ValueError`. Oversized models → `MemoryError` before allocation.

Default device discovery prefers a GPU, then a CPU; unusable platforms are
skipped with a warning. `IGNORE_GPU` is a comma-separated device/vendor filter
(if it excludes every device, discovery falls back to an unfiltered device).
To select a specific device, supply a context; `PYOPENCL_CTX` is honored by
PyOpenCL's `create_some_context`, not by the solver's default discovery.

| Attribute / property | Notes |
|---|---|
| `Nx, Ny, Nz`, `dl`, `npml`, `dt`, `t`, `step_num` | Grid and time state |
| `dtype`, `real`, `complex_dtype` | Computation dtypes |
| `accum_fp64`, `accreal`, `accreal_complex_dtype` | Monitor accumulation precision |
| `device`, `ctx`, `queue` | OpenCL runtime |
| `Ex`…`Hz` | Host copies of device fields (`dtype`) |

| Method | Notes |
|---|---|
| `set_epsilon(eps_array, sigma_array=None, pec_mask=None)` | Cell-wise isotropic εᵣ (grid-shaped array-like, including tuples of 2-D slices) or diagonal anisotropic 3-tuple `(eps_x, eps_y, eps_z)` and optional ohmic σ (S/m), each shape `(Nx,Ny,Nz)`; εᵣ must be finite and ≥ 1 (nondispersive dielectric; the fixed 0.99 vacuum-based Courant factor would in any case be violated below 0.9801), and σ must be finite and nonnegative. Builds Yee-edge `Ca`, `Cb` coefficients. Each E-edge takes the arithmetic mean of εᵣ (and σ) over the four cells that own it, the cell with the same index and its predecessors along the two transverse axes (`Ex[i,j,k]` at `(i+½, j, k)` averages cells `(i, j−1..j, k−1..k)`); at the domain boundary only in-domain cells contribute. Optional `pec_mask` (bool, same shape) marks perfect-conductor cells: all 12 Yee E-edges on each masked cell's boundary get `Ca = Cb = 0`, pinning tangential E to zero exactly on the closed cell surface. Use it for metal surfaces whose skin depth is far below the grid step; a finite σ there would need the skin depth resolved. With `npml > 0`, a biaxial tuple (three distinct components in a cell) must not reach the PML normal to the axis of its intermediate component; see below |
| `add_source_Jx(z_src, Jx, i0=None, i1=None, j0=None, j1=None, *, rim_taper=False, rim_edge=0.8, rim_renorm=True)` | SI Jx (A/m²) on a constant-z Ex sheet: `Ex += -Cb*Jx`; lossless `Cb=dt/(ε₀ εᵣ)` |
| `add_source_Ex(…)` | **Deprecated** soft Ex add; prefer `add_source_Jx` |
| `step()` | One Yee step (H → sources → E → monitors) |
| `run(n_steps, progress_every=0)` | Loop `step` |
| `read_point(name, i, j, k)` | Single cell without full-field download (`name` in `Ex`…`Hz`) |
| `estimate_device_memory_bytes(shape, npml, dtype=np.float32)` | Solver buffers: 6 fields + 6 material coefficients, CPML profiles and face-local ψ; excludes optional monitors and driver overhead |
| `device_memory_budget_bytes(device)` | Usable device bytes after reserved headroom |

**Biaxial media and the PML.** A cell whose three diagonal components all
differ supports waves that a PML normal to the axis of the intermediate
component amplifies instead of absorbing; the run then grows without bound.
`set_epsilon` therefore raises `ValueError` when, in a cell of the PML slabs
normal to axis *a* (the outer `npml` cells on both sides), `eps_a` lies
strictly between the other two components. For `(eps_x, eps_y, eps_z) =
(6, 2, 1)` that is the y-normal PML only: the medium may fill the x- and
z-normal PMLs. Inside the affected PML set `eps_a` equal to one of the other
two components (vacuum, another isotropic medium, or a uniaxial one).
Isotropic and uniaxial media are unrestricted, as is any medium with
`npml=0`. A tuple of the wrong length is rejected when its entries are
grid-shaped; any other malformed tuple is reported as a shape mismatch.

Sheet indices must be integers: `0 <= z_src < Nz`, `0 <= i0 <= i1 <= Nx`,
`0 <= j0 <= j1 <= Ny`. Bounds are half-open; `None` means the corresponding
domain edge. Invalid indices raise `ValueError` before modifying fields;
empty sheets are no-ops. `Jx` must be finite. `rim_edge` must be finite and
nonnegative. Taper weights apply once per axis even on one-cell spans;
renormalization preserves the sum of current weights and rejects nonempty
sheets with all-zero weights.

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

Same Yee/CPML physics (including ohmic conductivity, PEC masks, and diagonal anisotropic εᵣ) and source/monitor control flow as OpenCL.
`set_epsilon(eps_array, sigma_array=None, pec_mask=None)` places εᵣ, σ, and PEC masks on Yee edges with the same four-owner-cell rules as OpenCL (shared `materials.yee_edge_*`). εᵣ values must be finite and ≥ 1; σ must be finite and nonnegative. Grid, material (including isotropic tuples of 2-D slices), and sheet validation match OpenCL. CPML ψ arrays are
**full volume** (`Nx×Ny×Nz` × 12). Optional `psi_dtype` (e.g. `float16`)
reduces ψ memory at the cost of precision; that dtype is retained across steps.

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

Each box face snaps to the nearest grid node and is then clamped to the grid.
A face exactly half-way between two nodes goes to the higher node, on both
sides of the box, so a box that spans a whole number of cells keeps that
extent wherever its centre falls (3 cells centred on a node become the 3
cells shifted up by half a cell). The constructor raises `ValueError` for a
`freq` that is not finite and positive, for a non-finite `center` or `size`,
for a `size` that is not positive along every axis, and for a box that
spans less than one cell along any axis after snapping and clamping.

| Method | Notes |
|---|---|
| `get_farfield(obs)` | Complex `(Ex,Ey,Ez,Hx,Hy,Hz)` for far-field observation vector `obs` in metres (direction `obs/\|obs\|`, range `\|obs\|`, phase referenced to the grid origin) |
| `get_farfields(points)` | Shape `(n,6)` for far-field observation vectors in metres (`OpenCLNear2FarMonitor`; same convention as `get_farfield`). `points` is `(n,3)` or a single `(3,)` vector; an empty list or `(0,3)` array returns shape `(0,6)`, any other shape raises `ValueError` |
| `farfield_polar_xz(*, distance_m=1000, n_angles=73)` | OpenCL XZ cut at fixed range `distance_m` along `(sin θ, 0, cos θ)` (equivalent to a sphere of radius `distance_m` centered on the box, since translation cancels in `\|S\|`). Returns angles in degrees from +z toward +x and absolute `10*log10(\|S\|)`, with zero mapped to −∞ |
| `snapshot_dft()` / `dft_relative_change()` | Device-side convergence helper (OpenCL) |
| `fetch_dft_fields()` | Download the face-packed DFT to the host (OpenCL, debug): returns the six face packs keyed `"Ex"` … `"Hz"` and also fills sparse host volumes `Ex_dft` … `Hz_dft` |

Both paths integrate **face-packed** tangential DFT samples; OpenCL stores them
on the device, NumPy on the host. Accumulation precision follows
`fdtd.accreal_complex_dtype` when available, otherwise the field precision.
`fetch_dft_fields` is an OpenCL debugging helper. The face packs it returns
(order x0, x1, y0, y1, z0, z1; complex `fdtd.accreal_complex_dtype`) are the
complete data. The sparse volumes place each sample at the cell index of its
face cell and are for inspection only: along the three box edges through the
low corner two faces share a cell index, and the volume keeps the sample of
the face written last (x faces, then y, then z).

DFTs use `exp(+i*omega*t)`; outgoing fields use `exp(+i*k*r)/r` and
`H = cross(rhat, E)/ETA0`. Choose observation points in the far zone and a
Huygens box in homogeneous vacuum, outside the scatterer and inside the PML
interface. Subtract the peak dB yourself when a normalized plot is desired.

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
- **Materials:** Isotropic or diagonal anisotropic nondispersive εᵣ, optional ohmic σ, and cell-centered `pec_mask` on both OpenCL and NumPy engines; μ=μ₀. Biaxial media must stay out of the PML normal to their intermediate-ε axis. No periodic/symmetry BCs in-core.
- **Stability / memory:** Prefer checking `estimate_device_memory_bytes` on large grids; the OpenCL constructor already enforces headroom.

Internal modules (`cpml`, `materials`, `kernels`, …) are implementation details and may change without notice.
