# Mid-plane Ex + near-to-far baselines (OpenCL | Meep)

Abstract cases only — no application-specific geometry.

## Mid-plane Ex

Sources use SI current density `Jx` on both sides (`OpenCLFDTD.add_source_Jx` →
`Ex += −dt/(ε₀εᵣ)J` with Meep-tuned rim taper `rim_edge=0.8` + ∫J renorm;
Meep `CustomSource` is `J` with `meep_jx_from_si(..., resolution=...)` so Meep’s
planar δ × `gv.a` cancels).

| Case | Features exercised |
|------|--------------------|
| `vacuum_sheet` | SI Jx sheet + rim taper (PML-trimmed), CPML, matched Courant, mid-plane Ex |
| `dielectric_block` | Same + centered ε=4 rectangular block |

**Evidence** (CI-enforced):

- [`DISCREPANCY_REPORT.md`](DISCREPANCY_REPORT.md)
- [`discrepancy_report.json`](discrepancy_report.json)

```bash
IGNORE_GPU=NVIDIA,AMD PYOPENCL_CTX=0 python -m tests.meep_validation.update_plane_baselines
```

## Near-to-far `|S|(θ)`

Compact Ex patch + Huygens box (same geometry as live `test_meep_validation` far-field cases).

| Case | Features exercised |
|------|--------------------|
| `vacuum_farfield` | N2F XZ pattern + EH null depth |
| `dielectric_sphere_farfield` | Same + centered ε=4 sphere |

**Evidence** (CI-enforced):

- [`DISCREPANCY_REPORT_FARFIELD.md`](DISCREPANCY_REPORT_FARFIELD.md)
- [`discrepancy_report_farfield.json`](discrepancy_report_farfield.json)

```bash
IGNORE_GPU=NVIDIA,AMD PYOPENCL_CTX=0 python -m tests.meep_validation.update_farfield_baselines
```

CI requires pixel-identical PNGs and identical discrepancy reports versus a fresh POCL run, plus quality-gate floors.
