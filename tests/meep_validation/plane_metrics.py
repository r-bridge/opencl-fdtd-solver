# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
#
# This file is part of opencl-fdtd-solver.

"""Objective OpenCL ↔ Meep mid-plane Ex discrepancy metrics and reports."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CheckpointDiscrepancy:
    step: int
    pearson_corr: float
    """Signed Pearson correlation after LMS amplitude scale of Meep onto OpenCL."""
    lms_scale: float
    """Least-squares scale: OpenCL ≈ scale · Meep on the non-PML mask."""
    raw_residual_energy_ratio: float
    """∑(O−M)² / ∑O² on the non-PML mask (no amplitude alignment)."""
    aligned_residual_energy_ratio: float
    """∑(O−scale·M)² / ∑O² after LMS scale."""
    peak_opencl: float
    peak_meep: float
    peak_ratio: float
    """peak|OpenCL| / peak|Meep| on the non-PML mask."""
    mid_x_lag_cells: int
    """Best cross-correlation lag of mid-x Ex lineouts (Meep relative to OpenCL)."""


def _pml_mask(shape: tuple[int, int], npml: int) -> np.ndarray:
    nx, nz = shape
    m = np.ones((nx, nz), dtype=bool)
    p = max(0, int(npml))
    if p > 0:
        m[:p, :] = False
        m[-p:, :] = False
        m[:, :p] = False
        m[:, -p:] = False
    return m


def _round_metric(x: float) -> float:
    """Stable serialization for golden JSON / Markdown tables."""
    return float(f"{float(x):.8g}")


def measure_checkpoint(
    ocl: np.ndarray,
    meep: np.ndarray,
    *,
    npml: int,
    step: int,
) -> CheckpointDiscrepancy:
    a = np.asarray(ocl, dtype=np.float64)
    b = np.asarray(meep, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch at step {step}: {a.shape} vs {b.shape}")
    mask = _pml_mask(a.shape, npml)
    av = a[mask].ravel()
    bv = b[mask].ravel()
    e_o = float(np.dot(av, av)) + 1e-30
    denom = float(np.dot(bv, bv))
    scale = float(np.dot(av, bv) / denom) if denom > 0.0 else 0.0
    bv_s = bv * scale
    raw_r = float(np.dot(av - bv, av - bv) / e_o)
    al_r = float(np.dot(av - bv_s, av - bv_s) / e_o)
    if float(np.std(av)) < 1e-15 or float(np.std(bv_s)) < 1e-15:
        corr = 0.0
    else:
        corr = float(np.corrcoef(av, bv_s)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
    peak_a = float(np.max(np.abs(av))) if av.size else 0.0
    peak_b = float(np.max(np.abs(bv))) if bv.size else 0.0
    peak_ratio = peak_a / peak_b if peak_b > 0.0 else float("inf")

    ix = a.shape[0] // 2
    zo = a[ix, npml : a.shape[1] - npml] if npml > 0 else a[ix]
    zm = b[ix, npml : b.shape[1] - npml] if npml > 0 else b[ix]
    if zo.size < 2 or float(np.std(zo)) < 1e-30 or float(np.std(zm)) < 1e-30:
        lag = 0
    else:
        za = (zo - zo.mean()) / (np.std(zo) + 1e-30)
        zb = (zm - zm.mean()) / (np.std(zm) + 1e-30)
        xc = np.correlate(za, zb, mode="full")
        lags = np.arange(-len(za) + 1, len(za))
        lag = int(lags[int(np.argmax(xc))])

    return CheckpointDiscrepancy(
        step=int(step),
        pearson_corr=_round_metric(corr),
        lms_scale=_round_metric(scale),
        raw_residual_energy_ratio=_round_metric(raw_r),
        aligned_residual_energy_ratio=_round_metric(al_r),
        peak_opencl=_round_metric(peak_a),
        peak_meep=_round_metric(peak_b),
        peak_ratio=_round_metric(peak_ratio),
        mid_x_lag_cells=int(lag),
    )


def measure_case(
    ocl: dict[int, np.ndarray],
    meep: dict[int, np.ndarray],
    *,
    npml: int,
    checkpoints: list[int] | tuple[int, ...],
) -> list[CheckpointDiscrepancy]:
    rows: list[CheckpointDiscrepancy] = []
    for step in checkpoints:
        rows.append(measure_checkpoint(ocl[step], meep[step], npml=npml, step=int(step)))
    return rows


def case_summary(rows: list[CheckpointDiscrepancy]) -> dict[str, float]:
    if not rows:
        return {
            "mean_pearson_corr": 0.0,
            "mean_lms_scale": 0.0,
            "mean_raw_residual_energy_ratio": 0.0,
            "mean_aligned_residual_energy_ratio": 0.0,
            "max_abs_mid_x_lag_cells": 0.0,
        }
    return {
        "mean_pearson_corr": _round_metric(float(np.mean([r.pearson_corr for r in rows]))),
        "mean_lms_scale": _round_metric(float(np.mean([r.lms_scale for r in rows]))),
        "mean_raw_residual_energy_ratio": _round_metric(
            float(np.mean([r.raw_residual_energy_ratio for r in rows]))
        ),
        "mean_aligned_residual_energy_ratio": _round_metric(
            float(np.mean([r.aligned_residual_energy_ratio for r in rows]))
        ),
        "max_abs_mid_x_lag_cells": float(max(abs(r.mid_x_lag_cells) for r in rows)),
    }


def build_discrepancy_document(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate machine-readable discrepancy document."""
    return {
        "title": "OpenCL ↔ Meep mid-plane Ex discrepancy report",
        "description": (
            "Objective comparison of mid-plane Ex between this OpenCL Yee/CPML "
            "solver and Meep on shared abstract cases (SI Jx sheet, matched "
            "Courant, sheet trimmed out of PML). Metrics use the non-PML mask."
        ),
        "metric_definitions": {
            "pearson_corr": (
                "Pearson correlation of signed Ex after least-squares scaling "
                "Meep onto OpenCL (shape/phase agreement; amplitude removed)."
            ),
            "lms_scale": (
                "Least-squares scale such that OpenCL ≈ scale · Meep. "
                "Near 1.0 means absolute amplitude agreement after source matching."
            ),
            "raw_residual_energy_ratio": (
                "∑(OpenCL−Meep)² / ∑OpenCL² without amplitude alignment."
            ),
            "aligned_residual_energy_ratio": (
                "∑(OpenCL−scale·Meep)² / ∑OpenCL² after LMS scale."
            ),
            "peak_ratio": "peak|OpenCL| / peak|Meep| on the non-PML mask.",
            "mid_x_lag_cells": (
                "Integer lag maximizing mid-x lineout cross-correlation "
                "(Meep shifted relative to OpenCL)."
            ),
        },
        "cases": cases,
    }


def discrepancy_markdown(doc: dict[str, Any]) -> str:
    lines = [
        f"# {doc['title']}",
        "",
        doc["description"],
        "",
        "## Metric definitions",
        "",
    ]
    for key, text in doc["metric_definitions"].items():
        lines.append(f"- **`{key}`:** {text}")
    lines.append("")

    for case in doc["cases"]:
        lines.extend(
            [
                f"## Case `{case['name']}`",
                "",
                (
                    f"- Grid `{tuple(case['shape'])}`, dl={case['dl_m']} m, "
                    f"npml={case['npml']}, n_steps={case['n_steps']}"
                ),
                (
                    f"- Source freq={case['freq_hz']} Hz, fwidth={case['fwidth_hz']} Hz; "
                    f"Courant={case['courant']}"
                ),
                (
                    f"- Dielectric block half-cells={tuple(case['block_half'])}, "
                    f"ε={case['block_eps']}"
                ),
                "",
                "### Summary",
                "",
                f"| mean corr | mean LMS scale | mean raw res/E | mean aligned res/E | max \\|lag\\| |",
                f"|----------:|---------------:|---------------:|-------------------:|----------:|",
            ]
        )
        s = case["summary"]
        lines.append(
            f"| {s['mean_pearson_corr']:.6f} | {s['mean_lms_scale']:.6f} | "
            f"{s['mean_raw_residual_energy_ratio']:.6f} | "
            f"{s['mean_aligned_residual_energy_ratio']:.6f} | "
            f"{int(s['max_abs_mid_x_lag_cells'])} |"
        )
        lines.extend(
            [
                "",
                "### Checkpoints",
                "",
                "| step | corr | LMS scale | raw res/E | aligned res/E | peak_ocl | peak_meep | peak ratio | lag |",
                "|-----:|-----:|----------:|----------:|--------------:|---------:|----------:|-----------:|----:|",
            ]
        )
        for row in case["checkpoints"]:
            lines.append(
                f"| {row['step']} | {row['pearson_corr']:.6f} | {row['lms_scale']:.6f} | "
                f"{row['raw_residual_energy_ratio']:.6f} | "
                f"{row['aligned_residual_energy_ratio']:.6f} | "
                f"{row['peak_opencl']:.6e} | {row['peak_meep']:.6e} | "
                f"{row['peak_ratio']:.6f} | {row['mid_x_lag_cells']} |"
            )
        lines.append("")
        if case.get("images"):
            lines.append("### Images")
            lines.append("")
            for img in case["images"]:
                lines.append(f"- [`{img}`]({case['name']}/{img})")
            lines.append("")
    lines.append(
        "_Generated by `python -m tests.meep_validation.update_plane_baselines`. "
        "CI requires this file and `discrepancy_report.json` to match a fresh run._"
    )
    lines.append("")
    return "\n".join(lines)


def write_discrepancy_reports(root: Path, doc: dict[str, Any]) -> tuple[Path, Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "discrepancy_report.json"
    md_path = root / "DISCREPANCY_REPORT.md"
    json_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(discrepancy_markdown(doc), encoding="utf-8")
    return json_path, md_path


def case_report_dict(
    *,
    name: str,
    shape: list[int],
    dl_m: float,
    npml: int,
    n_steps: int,
    freq_hz: float,
    fwidth_hz: float,
    block_half: list[int],
    block_eps: float,
    courant: float,
    rows: list[CheckpointDiscrepancy],
    images: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(shape),
        "dl_m": dl_m,
        "npml": npml,
        "n_steps": n_steps,
        "freq_hz": freq_hz,
        "fwidth_hz": fwidth_hz,
        "block_half": list(block_half),
        "block_eps": block_eps,
        "courant": courant,
        "summary": case_summary(rows),
        "checkpoints": [asdict(r) for r in rows],
        "images": list(images),
    }
