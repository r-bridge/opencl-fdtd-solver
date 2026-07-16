# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
#
# This file is part of opencl-fdtd-solver.

"""Near-to-far golden cases wrapping cases.py far-field runners."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from opencl_fdtd_solver.monitors import ETA0

from .cases import (
    DL,
    FREQ,
    FWIDTH,
    N_STEPS,
    NPML,
    SHAPE,
    _sphere_eps,
    eh_from_list,
    run_meep_farfield_pattern,
    run_opencl_farfield_pattern,
)
from .farfield_metrics import (
    build_farfield_discrepancy_document,
    case_farfield_report_dict,
    measure_farfield_case,
    write_farfield_discrepancy_reports,
)
from .farfield_render import write_pattern_overlay_png
from .harness import OPENCL_COURANT
from .plane_cases import baselines_root

N_ANGLES = 19


@dataclass(frozen=True)
class FarfieldCase:
    """Abstract near-to-far consistency case (compact Ex patch + Huygens box)."""

    name: str
    # None ⇒ vacuum; else sphere ε_r and OpenCL radial extent in cells.
    sphere_eps: float | None = None
    sphere_rad_cells: int = 6
    # Main-lobe |Δ|dB gate (matches test_meep_validation).
    max_main_lobe_db: float = 2.5

    @property
    def has_dielectric(self) -> bool:
        return self.sphere_eps is not None and self.sphere_eps > 1.0 + 1e-12


def default_farfield_cases() -> list[FarfieldCase]:
    return [
        FarfieldCase(name="vacuum_farfield", max_main_lobe_db=2.5),
        FarfieldCase(
            name="dielectric_sphere_farfield",
            sphere_eps=4.0,
            sphere_rad_cells=6,
            max_main_lobe_db=3.0,
        ),
    ]


def _normalize_meep_eh(eh: np.ndarray) -> np.ndarray:
    """Convert Meep H (ε₀≡1 units) to SI-comparable amplitudes (÷ η₀)."""
    out = np.asarray(eh, dtype=np.complex128).copy()
    out[3:6] = out[3:6] / float(ETA0)
    return out


def run_opencl_farfield_case(case: FarfieldCase) -> dict[str, Any]:
    eps = None
    if case.has_dielectric:
        eps = _sphere_eps(float(case.sphere_eps), rad_cells=int(case.sphere_rad_cells))
    raw = run_opencl_farfield_pattern(n_angles=N_ANGLES, eps=eps)
    return {
        "angles_deg": np.asarray(raw["angles_deg"], dtype=np.float64),
        "S_db": np.asarray(raw["S_db"], dtype=np.float64),
        "eh_plus_z": eh_from_list(raw["eh_plus_z"]),
        "eh_plus_x": eh_from_list(raw["eh_plus_x"]),
        "obs_r_m": float(raw["obs_r_m"]),
    }


def run_meep_farfield_case(case: FarfieldCase) -> dict[str, Any]:
    eps_sphere = float(case.sphere_eps) if case.has_dielectric else None
    raw = run_meep_farfield_pattern(n_angles=N_ANGLES, eps_sphere=eps_sphere)
    return {
        "angles_deg": np.asarray(raw["angles_deg"], dtype=np.float64),
        "S_db": np.asarray(raw["S_db"], dtype=np.float64),
        "eh_plus_z": _normalize_meep_eh(eh_from_list(raw["eh_plus_z"])),
        "eh_plus_x": _normalize_meep_eh(eh_from_list(raw["eh_plus_x"])),
        "obs_r_m": None,
    }


def run_case_farfields(case: FarfieldCase) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_opencl_farfield_case(case), run_meep_farfield_case(case)


def load_case_farfields_from_baselines(case_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    case_dir = Path(case_dir)
    ocl = {
        "angles_deg": np.load(case_dir / "angles_deg.npy"),
        "S_db": np.load(case_dir / "ocl_S_db.npy"),
        "eh_plus_z": np.load(case_dir / "ocl_eh_plus_z.npy"),
        "eh_plus_x": np.load(case_dir / "ocl_eh_plus_x.npy"),
    }
    meep = {
        "angles_deg": np.load(case_dir / "angles_deg.npy"),
        "S_db": np.load(case_dir / "meep_S_db.npy"),
        "eh_plus_z": np.load(case_dir / "meep_eh_plus_z.npy"),
        "eh_plus_x": np.load(case_dir / "meep_eh_plus_x.npy"),
    }
    return ocl, meep


def write_case_farfield_baselines(
    case: FarfieldCase,
    out_dir: str | Path,
    *,
    ocl: dict[str, Any] | None = None,
    meep: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write pattern PNG + arrays + meta.json under ``out_dir/<case.name>/``."""
    root = Path(out_dir)
    dest = root / case.name
    dest.mkdir(parents=True, exist_ok=True)
    if ocl is None or meep is None:
        ocl, meep = run_case_farfields(case)
    assert ocl is not None and meep is not None

    np.testing.assert_allclose(ocl["angles_deg"], meep["angles_deg"], rtol=0, atol=1e-12)
    angles = np.asarray(ocl["angles_deg"], dtype=np.float64)
    ocl_db = np.asarray(ocl["S_db"], dtype=np.float64)
    meep_db = np.asarray(meep["S_db"], dtype=np.float64)

    metrics = measure_farfield_case(ocl_db, meep_db, ocl, meep)
    png_name = "pattern_xz.png"
    write_pattern_overlay_png(dest / png_name, angles, ocl_db, meep_db)

    np.save(dest / "angles_deg.npy", angles)
    np.save(dest / "ocl_S_db.npy", ocl_db.astype(np.float64))
    np.save(dest / "meep_S_db.npy", meep_db.astype(np.float64))
    np.save(dest / "ocl_eh_plus_z.npy", np.asarray(ocl["eh_plus_z"], dtype=np.complex128))
    np.save(dest / "meep_eh_plus_z.npy", np.asarray(meep["eh_plus_z"], dtype=np.complex128))
    np.save(dest / "ocl_eh_plus_x.npy", np.asarray(ocl["eh_plus_x"], dtype=np.complex128))
    np.save(dest / "meep_eh_plus_x.npy", np.asarray(meep["eh_plus_x"], dtype=np.complex128))

    meta: dict[str, Any] = {
        "name": case.name,
        "shape": list(SHAPE),
        "dl_m": DL,
        "npml": NPML,
        "n_steps": N_STEPS,
        "freq_hz": FREQ,
        "fwidth_hz": FWIDTH,
        "n_angles": N_ANGLES,
        "sphere_eps": case.sphere_eps,
        "sphere_rad_cells": case.sphere_rad_cells if case.has_dielectric else 0,
        "max_main_lobe_db": case.max_main_lobe_db,
        "courant": float(OPENCL_COURANT),
        "source": "soft_Ex_compact_patch",
        "files": [png_name],
        "discrepancy": metrics,
    }
    (dest / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = case_farfield_report_dict(
        name=case.name,
        shape=list(SHAPE),
        dl_m=DL,
        npml=NPML,
        n_steps=N_STEPS,
        freq_hz=FREQ,
        fwidth_hz=FWIDTH,
        n_angles=N_ANGLES,
        sphere_eps=case.sphere_eps,
        sphere_rad_cells=case.sphere_rad_cells if case.has_dielectric else 0,
        max_main_lobe_db=case.max_main_lobe_db,
        courant=float(OPENCL_COURANT),
        metrics=metrics,
        images=[png_name],
    )
    return dest, report


def write_all_farfield_baselines(
    out_dir: str | Path | None = None,
    cases: list[FarfieldCase] | None = None,
) -> Path:
    root = baselines_root() if out_dir is None else Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if cases is None:
        cases = default_farfield_cases()
    case_docs: list[dict[str, Any]] = []
    for case in cases:
        _dest, report = write_case_farfield_baselines(case, root)
        case_docs.append(report)
    doc = build_farfield_discrepancy_document(case_docs)
    write_farfield_discrepancy_reports(root, doc)
    return root
