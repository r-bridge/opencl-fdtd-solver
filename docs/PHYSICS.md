# Physics formulation and validation guide

This document is for physicists reviewing whether the solver’s electromagnetics are sound. It describes the equations that are discretized, the assumptions that bound the model, and the independent checks used to validate correctness. Implementation detail is mentioned only when it affects the physics.

For the public Python API (constructors, sources, monitors), see [`API.md`](API.md).

---

## 1. What this solver is

A **second-order Yee finite-difference time-domain (FDTD)** solver in **SI units**, with **complex-frequency-shifted convolutional PML (CFS-CPML)** on the outer boundary, optional **soft current sources**, and optional **near-to-far (Huygens) transformation**.

It is intentionally a **kernel**, not a full CAD/EM suite:

| Included | Not included |
|---|---|
| Uniform cubic Yee grid | Non-uniform / unstructured meshes |
| Nondispersive scalar εᵣ, optional ohmic σ on OpenCL | Lorentz / Debye / Drude media |
| Vacuum μ = μ₀ | Magnetic materials (μᵣ ≠ 1) |
| Soft Ex / SI Jx sheet sources | Built-in antennas, PEC/PMC, periodic BCs |
| Closed-box Huygens near-to-far | Full geometric optics / high-frequency asymptotics |
| OpenCL FP32 (default) and FP64 (`dtype=float64`) | Separate CUDA backend / second kernel dialect |

For the supported physics on a matched grid, the accuracy class is that of **default Meep FDTD** (same order, same leapfrog). Throughput is the design goal, not higher-order fidelity.

---

## 2. Units and constants

Everything in the solver core is SI:

| Quantity | Unit |
|---|---|
| Length (`dl`, observation points) | m |
| Time (`dt`, `t`) | s |
| Frequency | Hz |
| Electric field **E** | V/m |
| Magnetic field **H** | A/m |
| Current density **J** | A/m² |

Physical constants (`opencl_fdtd_solver/constants.py`):

\[
c_0 = 299\,792\,458\,\mathrm{m/s},\quad
\mu_0 = 4\pi\times 10^{-7}\,\mathrm{H/m},\quad
\varepsilon_0 = 1/(\mu_0 c_0^2),\quad
\eta_0 = \sqrt{\mu_0/\varepsilon_0}\approx 376.73\,\Omega.
\]

Meep validation scripts may use a millimetre length unit internally; that conversion lives only in the test harness, not in the solver.

---

## 3. Continuous Maxwell system (what is approximated)

In vacuum or nondispersive dielectric with **μ = μ₀** and no free charges:

\[
\nabla\times\mathbf{E} = -\mu_0\,\partial_t\mathbf{H},\qquad
\nabla\times\mathbf{H} = \varepsilon_0\varepsilon_r\,\partial_t\mathbf{E} + \sigma\mathbf{E} + \mathbf{J}.
\]

The OpenCL engine supports scalar, frequency-independent **ohmic conductivity**
σ. The NumPy reference is lossless (σ=0). Neither supports magnetic contrast or
dispersive polarization models.

---

## 4. Yee discretization

### 4.1 Grid

- Domain: \(N_x\times N_y\times N_z\) cells of size \(\Delta\ell\) (called `dl`).
- Cell-centered material: \(\varepsilon_r(i,j,k)\).
- Standard Yee staggering: **E** on edges, **H** on faces. Explicitly:
  - \(E_x\) at \((i+\tfrac12,j,k)\), \(E_y\) at \((i,j+\tfrac12,k)\), \(E_z\) at \((i,j,k+\tfrac12)\).
- Edge ε used in the E update is the **arithmetic mean** of the two cells sharing that edge (last plane along the stagger axis keeps the cell value).

### 4.2 Leapfrog update order

Each time step:

1. Advance **H** to time \(n+\tfrac12\).
2. Sample current sources at time \(n+\tfrac12\) (half-step).
3. Advance **E** to time \(n+1\), including the current term after old-field decay.
4. Advance \(t\leftarrow t+\Delta t\); optionally accumulate DFT / near-to-far monitors.

Interior curls (before PML terms) are the usual centered differences:

\[
\mathbf{H}^{n+1/2}
  \leftarrow
  \mathbf{H}^{n-1/2}
  - \frac{\Delta t}{\mu_0}\,\nabla\times\mathbf{E}^{n},
\]

\[
\mathbf{E}^{n+1}
  \leftarrow
  C_a\mathbf{E}^{n}
  + C_b\bigl(\nabla\times\mathbf{H}^{n+1/2}-\mathbf{J}^{n+1/2}\bigr),
\qquad
C_a=\frac{1-q}{1+q},\quad
C_b=\frac{\Delta t/\varepsilon}{1+q},\quad
q=\frac{\sigma\Delta t}{2\varepsilon},\quad
\varepsilon=\varepsilon_0\varepsilon_r.
\]

### 4.3 Courant number

Default time step (3-D CFL with safety factor 0.99):

\[
S = \frac{0.99}{\sqrt{3}}\approx 0.5716,\qquad
\Delta t = S\,\frac{\Delta\ell}{c_0} = 0.99\,\frac{\Delta\ell}{c_0\sqrt{3}}.
\]

Meep comparisons force the **same** Courant (Meep’s own default is 0.5). Changing \(S\) without matching the reference invalidates amplitude/phase comparisons.

### 4.4 Numerical dispersion (analytic check)

For a plane wave along a lattice axis, the discrete dispersion relation used in the analytic suite is

\[
\sin\!\Bigl(\frac{\omega\Delta t}{2}\Bigr)
=
S\,\sin\!\Bigl(\frac{\tilde{k}\,\Delta\ell}{2}\Bigr).
\]

This is the expected Yee dispersion; there is no higher-order correction. The
regression propagates a pulse, measures phase at two probes, and checks both
this relation and decreasing phase-velocity error on grid refinement.

---

## 5. Materials

**Relative permittivity** \(\varepsilon_r\) and optional OpenCL **conductivity**
\(\sigma\) are programmable:

- Stored as a cell-centered array via `set_epsilon`.
- Both are averaged onto Yee edges and converted to the trapezoidal-loss
  coefficients \(C_a,C_b\) above.
- Default: vacuum \(\varepsilon_r=1,\sigma=0\).

**Implications for reviewers**

- Interfaces are voxel staircased unless you pre-average ε yourself.
- Subpixel / effective-medium construction is **application responsibility**.
- Dielectric contrast is correct in the Yee sense for the supplied voxels; it is not a CAD solid model.

---

## 6. Absorbing boundary: CFS-CPML

Outer layers use **complex-frequency-shifted convolutional PML** with polynomial grading (Taflove-style parameters in `opencl_fdtd_solver/cpml.py`):

| Parameter | Value |
|---|---|
| Thickness | `npml` cells (constructor default **20**; validation cases often 4–12) |
| Grading order \(m\) | 3 |
| \(\sigma_{\max}\) | \(0.8\,(m+1)/(\eta_0\Delta\ell)\) |
| \(\kappa_{\max}\) | 15 |
| \(\alpha_{\max}\) | \(0.05/\eta_0\) at the vacuum interface, decaying toward the outer wall |

Separate 1-D profiles are built for E-node and H-node locations (half-cell stagger). Auxiliary ψ fields implement the CFS convolution; the OpenCL engine stores them on **PML face slabs** only (not the full volume).

**What to expect**

- Normal-incidence reflection is gated analytically below about **−25 dB** in the unit suite.
- Thin PML (e.g. 4 cells on tiny validation grids) is a compromise for CI cost, not a claim of ultra-low reflection for all angles/spectra.
- Energy after the source turns off should decay; Meep-matched PML decay tests require late/peak Ex energy \(\lt 0.05\).

---

## 7. Sources

### 7.1 Preferred: SI soft current (Jx)

On a constant-\(z\) \(E_x\) sheet:

\[
E_x \;\mathrel{+}=\; -C_b\,J_x\,w
\]

with \(J_x\) in A/m² and \(C_b=\Delta t/(\varepsilon_0\varepsilon_r)\) in a
lossless medium. This matches Meep’s \(D\leftarrow D-J\,\Delta t\) once SI
\(\varepsilon_0\) is restored. In a conductor the source is not multiplied by
an additional \(C_a\); a driven, uniform Ampère-law recurrence checks this.

Optional **rim taper** (default rim weight 0.8) down-weights the sheet perimeter (edges ×0.8, corners ×0.64) to better match Meep’s continuous volume-source restriction. With renorm enabled, the weights are rescaled so the **net ∫J** is preserved.

Sheets used in Meep baselines are trimmed **out of the PML**.

### 7.2 Legacy: soft Ex add

`Ex += amplitude` injects field directly. Prefer Jx for absolute SI comparisons.

### 7.3 Timing

Current sources are sampled at the **H half-step** (\(n+\tfrac12\)), consistent
with leapfrog staggering, and added to the completed E update. Legacy field
adds are immediate and, if applied before that update, undergo its decay.

---

## 8. Near-to-far transformation

A closed rectangular **Huygens surface** is defined in physical metres and snapped to Yee indices. On each face:

\[
\mathbf{J}_s = \hat{n}\times\mathbf{H},\qquad
\mathbf{M}_s = -\hat{n}\times\mathbf{E}.
\]

Face DFT uses \(\int f(t)e^{+i\omega t}\,dt\). **H** carries the half-step
phase \(e^{-i\omega\Delta t/2}\) so E and H are co-located in time for the
integral. Radiation integrals \(\mathbf{N},\mathbf{L}\) of the currents above
use phase \(e^{-ik\hat r\cdot\mathbf{r}'}\) and trapezoidal edge/corner weights.

Far fields with this Fourier convention (outgoing \(e^{+ikr}\)):

\[
\mathbf{E}
=
\frac{ik}{4\pi r}\,e^{+ikr}
\bigl(\eta_0\,\mathbf{N}_\perp - \hat{r}\times\mathbf{L}\bigr),\qquad
\mathbf{H}
=
\frac{1}{\eta_0}\,\hat{r}\times\mathbf{E}.
\]

Internally the implementation integrates the negatives of these conventional
currents (\(\mathbf{H}\times\hat n\), \(\hat n\times\mathbf{E}\)), so its
prefactor is \(-ik\), giving the same fields. The positive H relation ensures
outward time-averaged energy flux. Radius and source-translation regressions
check phase, not just magnitudes.

Pattern plots use complex Poynting magnitude; power dB is
\(10\log_{10}|S|\). The plotting comparisons subtract the peak; the API returns
unnormalized power dB. For an \(E_x\)-driven sheet, expect a main lobe along
\(\pm z\) with \(E_x/H_y\) polarization and a deep endfire null along \(\pm x\).

**Caveat for reviewers:** null floors vs Meep can differ by orders of magnitude on these tiny grids; main-lobe comparisons within ~12 dB of peak are the intended metric, not absolute null depth.

---

## 9. Floating-point precision

The default OpenCL path is **float32**. Optional **`dtype=np.float64`** rebuilds the same kernels with `real=double` when the device exposes `cl_khr_fp64` (and `cl_khr_int64_base_atomics` for near-to-far). There is no parallel CUDA dialect — FP64 is the same OpenCL sources.

Consequences:

- FP32: dynamic range and late-time cancellation are weaker than double-precision Meep; absolute amplitudes can disagree by ~10–20% while **shapes** remain highly correlated after LMS scaling (see baselines below).
- FP64: expect roughly **~2×** slower wall time on bandwidth-bound grids (bytes/cell double); use it for long coherent runs or deep nulls when the device supports it.
- Device without FP64 extensions: constructing `OpenCLFDTD(..., dtype=float64)` fails clearly.

---

## 10. How soundness is validated

Three independent layers. You do not need to read the OpenCL code to interpret them.

### 10.1 Analytic / self-consistency (no Meep)

Implemented in `tests/test_analytic_validation.py`:

- Yee dispersion relation (plane-wave phase).
- Far-field angular shape for a compact \(E_x\) source \(\sim\cos^2\theta\) in the XZ cut, plus a strict endfire null-depth gate.
- CPML normal-incidence reflection bound (\(\lesssim -25\,\mathrm{dB}\)).
- Closed-box (npml=0) electromagnetic energy stability after the source is off.
- Dielectric-sphere bistatic E-plane shape vs Bohren–Huffman Mie series (`tests/analytic_mie.py`; normalized pattern correlation, not absolute RCS).

`tests/test_correctness_regressions.py` additionally checks conductive-source
amplitudes, outward flux, one-sided Huygens radiation on all six faces,
radial/translation phase, and CPU/GPU agreement relative to field magnitude.

### 10.1.1 Metallic mirror: absolute RCS

`tests/test_mirror_rcs.py` runs an ideal, zero-thickness conducting square of
side \(a=3\lambda=0.18\) m under normal, x-polarized plane-wave illumination.
The test fixture imposes \(E_{s,t}=-E_{i,t}\) on tangential Yee edges; the
ordinary Yee/CPML kernels evolve the scattered field elsewhere. No second
solver or fitted amplitude scale is used. This fixture does not add a public
PEC API and does not validate finite-metal skin-depth meshing.

The measured quantity is \(\sigma=4\pi r^2|E_s|^2/|E_i|^2\), in m². The
large-plate physical-optics prediction for broadside backscatter is
\(\sigma_{\rm PO}=4\pi a^4/\lambda^2\approx3.66435\) m². The bistatic
E-plane prediction, at observation angle θ from the reflected normal, is
\(\sigma_{\rm PO}(\theta)=\sigma_{\rm PO}(0)\cos^2\theta
[\sin u/u]^2\), \(u=\pi a\sin\theta/\lambda\).

Gates: fine-grid broadside error <10%; absolute non-null angular samples
within 15%; first-null power <1% of broadside; end-of-run field change <0.5%.
Refining from 10 to 14 cells/λ must reduce the broadside discrepancy. These
tolerances include finite-grid and physical-optics approximation error:
the formula is **not an exact finite-plate Maxwell solution**, and edge
diffraction can fill its nulls. See the [NASA flat-plate RCS study](https://ntrs.nasa.gov/citations/19750002834).

### 10.2 OpenCL ↔ NumPy parity

The CPU NumPy engine implements the **same** Yee + CPML update. Field components after identical sources must match within tight absolute tolerances (`tests/test_solver.py`). This catches kernel bugs independent of Meep.

### 10.3 External reference: Meep

CI compares against [Meep](https://meep.readthedocs.io/) on shared abstract cases (matched Courant, SI Jx with rim taper, hard voxels with `eps_averaging=False` on the Meep side).

**Live gates** (see README §5):

| Check | Tolerance |
|---|---|
| Near-field Ex DFT (peak-normalized) | max error \(\lt 0.25\) |
| Far-field \(\lvert S\rvert(\theta)\) vacuum, main lobe (−12 dB mask) | \(\lt 2.5\,\mathrm{dB}\) |
| Polarization on \(+z\); null on \(+x\) | pol. error \(\lt 0.35\); \(\lvert E(+x)\rvert/\lvert E(+z)\rvert\lt 0.05\) |
| Dielectric sphere \(\varepsilon_r=4\) pattern | main lobe \(\lt 3\,\mathrm{dB}\) |
| PML energy decay | late/peak \(\lt 0.05\) |

Pattern gates use the **union** of angles within 12 dB of either peak so a
missing lobe cannot be masked away. Vector checks use one common complex
alignment for all six components, with H scaled by η₀, and require outward
flux; E and H cannot independently hide a sign error.

**Committed quantitative evidence** (regenerated and bit-compared in CI):

- Mid-plane Ex: [`tests/meep_validation/baselines/DISCREPANCY_REPORT.md`](../tests/meep_validation/baselines/DISCREPANCY_REPORT.md)  
  Includes Pearson correlation, aligned residual energy, and spatial lag.
- Far field: [`tests/meep_validation/baselines/DISCREPANCY_REPORT_FARFIELD.md`](../tests/meep_validation/baselines/DISCREPANCY_REPORT_FARFIELD.md)  
  Includes power-pattern discrepancy and null depth for vacuum and a dielectric sphere.

These baselines demonstrate agreement for the tested cases, not correctness
for every supported configuration or certification of a device design.

---

## 11. Checklist for a physics review

Use this when deciding whether results are trustworthy for your problem.

1. **Is my problem inside the model?** Scalar nondispersive ε, μ=μ₀, no PEC/periodic BCs, uniform Δℓ; FP32 enough or need `dtype=float64`?
2. **Resolution:** \(\Delta\ell\) small enough for the shortest wavelength *inside* the highest ε (rule of thumb: ≥10–20 cells/λ).
3. **Courant:** leave at default \(0.99/\sqrt{3}\) unless you knowingly rematch any reference.
4. **Materials:** supply a cell-wise εᵣ you trust (average subpixels yourself if needed).
5. **Sources:** prefer SI `Jx` sheets trimmed out of PML; use rim taper when matching volume-restricted references.
6. **PML:** `npml` thick enough for your spectrum and angles (default 20 is safer than the thin CI cases).
7. **Near-to-far:** Huygens box in the near field but outside scatterers; interpret main lobe, not absolute null floors, on coarse grids.
8. **Cross-check:** for a new configuration, compare a mid-plane cut or probe DFT against Meep (or the NumPy engine) before trusting far-field numbers.

---

## 12. When to use another solver

Prefer Meep or a commercial code if you need any of:

- dispersive / magnetic media,
- PEC, periodic, or symmetry boundaries,
- built-in geometry, subpixel averaging, or eigenmode ports,
- certification of a specific fabricated device without your own meshing pipeline.

---

## 13. References (formulation lineage)

- K. S. Yee, “Numerical solution of initial boundary value problems involving Maxwell’s equations in isotropic media,” *IEEE Trans. Antennas Propag.*, 1966.
- A. Taflove & S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*.
- J. A. Roden & S. D. Gedney, “Convolution PML (CPML),” *Microwave Optical Technol. Lett.*, 2000.
- C. A. Balanis, *Antenna Theory* (far-field equivalence / N, L integrals).
- Implementation heritage: [gprMax](https://github.com/gprMax/gprMax) Yee/CPML formulations (University of Edinburgh), stripped to this Python/OpenCL kernel.
- External cross-check: [Meep](https://meep.readthedocs.io/).

---

## 14. Related files in this repository

| Path | Role |
|---|---|
| `opencl_fdtd_solver/constants.py` | SI constants |
| `opencl_fdtd_solver/cpml.py` | CPML profile formulas |
| `opencl_fdtd_solver/materials.py` | Yee-edge ε averaging, \(c_e\) |
| `opencl_fdtd_solver/kernels/yee_update.cl` | Discrete H/E updates |
| `opencl_fdtd_solver/kernels/sources.cl` | Soft Ex / Jx |
| `opencl_fdtd_solver/kernels/dft_farfield.cl` | Face DFT + far-field reduce |
| `opencl_fdtd_solver/monitors.py` | Host near-to-far (readable twin of the kernels) |
| `tests/test_analytic_validation.py` | Dispersion, dipole pattern/null, PML \|R\|, energy, Mie shape |
| `tests/meep_validation/baselines/` | Committed Meep discrepancy reports |
