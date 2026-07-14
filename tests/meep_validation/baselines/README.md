# Mid-plane Ex baselines (OpenCL | Meep | residual)

Abstract cases only — no application-specific geometry.

Sources use SI current density `Jx` on both sides (`OpenCLFDTD.add_source_Jx` →
`Ex += −dt/(ε₀εᵣ)J` with Meep-tuned rim taper `rim_edge=0.8` + ∫J renorm;
Meep `CustomSource` is `J` with `meep_jx_from_si(..., resolution=...)` so Meep’s
planar δ × `gv.a` cancels).

| Case | Features exercised |
|------|--------------------|
| `vacuum_sheet` | SI Jx sheet + rim taper (PML-trimmed), CPML, matched Courant, mid-plane Ex |
| `dielectric_block` | Same + centered ε=4 rectangular block |

**Human / machine discrepancy evidence** (CI-enforced):

- [`DISCREPANCY_REPORT.md`](DISCREPANCY_REPORT.md) — tables of Pearson corr, LMS scale, residual energy ratios, peak ratios, mid-x lag
- [`discrepancy_report.json`](discrepancy_report.json) — same metrics as exact JSON

```bash
IGNORE_GPU=NVIDIA,AMD PYOPENCL_CTX=0 python -m tests.meep_validation.update_plane_baselines
```

CI requires pixel-identical PNGs and an identical discrepancy report versus a fresh POCL run, plus quality-gate floors on mean correlation / scale / residual.
