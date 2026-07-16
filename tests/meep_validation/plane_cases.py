# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
#
# This file is part of opencl-fdtd-solver.

"""Generic OpenCL ↔ Meep mid-plane Ex cases (no application-specific geometry).

Each case exercises features used by typical 3D FDTD workflows:
vacuum or a simple dielectric block, CPML, SI Jx sheet with Meep-matched rim
taper (trimmed out of PML), matched Courant, and mid-plane Ex sampling at fixed
checkpoints.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from opencl_fdtd_solver import OpenCLFDTD

from .harness import (
    C0,
    OPENCL_COURANT,
    ensure_pyopencl_ctx,
    gaussian_sine_amp,
    meep_jx_from_si,
    meep_until,
)
from .plane_render import write_triptych_png


@dataclass(frozen=True)
class PlaneCase:
    """Abstract mid-plane consistency case."""

    name: str
    shape: tuple[int, int, int]
    dl: float  # metres
    npml: int
    n_steps: int
    freq: float
    fwidth: float
    checkpoints: tuple[int, ...]
    # Centered dielectric block half-sizes in cells (0 ⇒ vacuum).
    block_half: tuple[int, int, int] = (0, 0, 0)
    block_eps: float = 1.0

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        dl_mm = self.dl * 1e3
        return tuple(n * dl_mm for n in self.shape)  # type: ignore[return-value]

    @property
    def dl_mm(self) -> float:
        return float(self.dl * 1e3)


def default_plane_cases() -> list[PlaneCase]:
    """Canonical golden cases (kept small for CI wall time).

    ``Nz != Nx`` so Meep ``get_array`` shapes distinguish ``(nx, nz)`` vs ``(nz, nx)``.
    """
    return [
        PlaneCase(
            name="vacuum_sheet",
            shape=(32, 32, 40),
            dl=2.5e-3,
            npml=4,
            n_steps=80,
            freq=5e9,
            fwidth=1e9,
            checkpoints=(20, 40, 60, 80),
        ),
        PlaneCase(
            name="dielectric_block",
            shape=(32, 32, 40),
            dl=2.5e-3,
            npml=4,
            n_steps=80,
            freq=5e9,
            fwidth=1e9,
            checkpoints=(20, 40, 60, 80),
            block_half=(6, 6, 4),
            block_eps=4.0,
        ),
    ]


def baselines_root() -> Path:
    return Path(__file__).resolve().parent / "baselines"


def _make_eps(case: PlaneCase) -> np.ndarray:
    eps = np.ones(case.shape, dtype=np.float32)
    hx, hy, hz = case.block_half
    if hx <= 0 and hy <= 0 and hz <= 0:
        return eps
    nx, ny, nz = case.shape
    ci, cj, ck = nx // 2, ny // 2, nz // 2
    eps[
        ci - hx : ci + hx + 1,
        cj - hy : cj + hy + 1,
        ck - hz : ck + hz + 1,
    ] = float(case.block_eps)
    return eps


def _source_z_index(case: PlaneCase) -> int:
    return int(case.shape[2] - case.npml - 2)


def run_opencl_planes(case: PlaneCase) -> dict[int, np.ndarray]:
    """OpenCL mid-plane Ex at ``y = Ny//2`` for each checkpoint."""
    ensure_pyopencl_ctx()
    nx, ny, nz = case.shape
    p = int(case.npml)
    z_src = _source_z_index(case)
    fdtd = OpenCLFDTD(case.shape, case.dl, npml=case.npml)
    fdtd.set_epsilon(_make_eps(case))

    def source_cb(f):
        f.add_source_Jx(
            z_src,
            gaussian_sine_amp(f.t, case.freq, case.fwidth),
            i0=p,
            i1=nx - p,
            j0=p,
            j1=ny - p,
            rim_taper=True,
        )

    fdtd.add_source(source_cb)
    want = set(case.checkpoints)
    out: dict[int, np.ndarray] = {}
    j_mid = ny // 2
    for step in range(1, case.n_steps + 1):
        fdtd.step()
        if step in want:
            plane = np.asarray(fdtd.Ex[:, j_mid, :], dtype=np.float64)
            out[step] = plane
    fdtd.queue.finish()
    return out


def _meep_script(case: PlaneCase, out_rel: str = ".") -> str:
    """Meep script writing ``meep_ex_stepXXXX.npy`` plus a JSON summary line."""
    nx, ny, nz = case.shape
    Lx, Ly, Lz = case.extent_mm
    dl_mm = case.dl_mm
    npml = case.npml
    z_src = _source_z_index(case)
    z_mm = (z_src + 0.5) * dl_mm - 0.5 * Lz
    pad = 2.0 * npml * dl_mm
    src_sx = max(dl_mm, Lx - pad)
    src_sy = max(dl_mm, Ly - pad)
    # Same SI Jx waveform as OpenCL; convert so Meep ΔE ≈ SI ΔE (ε₀≡1 in Meep),
    # including Meep's planar δ-source × resolution factor.
    j_unit = meep_jx_from_si(1.0, resolution=1.0 / dl_mm)
    hx, hy, hz = case.block_half
    block_mm = (
        (2 * hx + 1) * dl_mm,
        (2 * hy + 1) * dl_mm,
        (2 * hz + 1) * dl_mm,
    )
    if case.block_eps > 1.0 + 1e-12 and hx > 0:
        geom = f"""
geometry = [
    mp.Block(
        center=mp.Vector3(),
        size=mp.Vector3({block_mm[0]}, {block_mm[1]}, {block_mm[2]}),
        material=mp.Medium(epsilon={float(case.block_eps)}),
    )
]
"""
    else:
        geom = "geometry = []\n"

    checkpoints = list(case.checkpoints)
    lines_until = []
    t_prev = 0.0
    for step in checkpoints:
        t_abs = meep_until(step, case.dl)
        dt = max(t_abs - t_prev, 0.0)
        lines_until.append(
            f"sim.run(until={dt!r})\n"
            f"arr = np.asarray(sim.get_array(component=mp.Ex, vol=vol), dtype=np.float64)\n"
            f"# Meep XZ slice is (nz, nx); OpenCL mid-plane is (nx, nz).\n"
            f"if arr.ndim != 2:\n"
            f"    raise SystemExit(f'expected 2D Ex slice, got {{arr.shape}}')\n"
            f"if arr.shape == ({nz}, {nx}):\n"
            f"    arr = arr.T\n"
            f"elif arr.shape != ({nx}, {nz}):\n"
            f"    raise SystemExit(\n"
            f"        f'unexpected Ex slice shape {{arr.shape}}; want ({nx}, {nz}) or ({nz}, {nx})'\n"
            f"    )\n"
            f"np.save(f'{{out_dir}}/meep_ex_step{step:04d}.npy', arr)\n"
        )
        t_prev = t_abs

    body = "\n".join(lines_until)
    return f"""
import json
import os
import numpy as np
import meep as mp

out_dir = os.path.abspath({out_rel!r})
os.makedirs(out_dir, exist_ok=True)

cell = mp.Vector3({Lx}, {Ly}, {Lz})
resolution = {1.0 / dl_mm!r}
{geom}
freq_hz = {case.freq!r}
fwidth_hz = {case.fwidth!r}
t0 = 5.0 / (np.pi * fwidth_hz)
sigma = 1.0 / (np.pi * fwidth_hz)
# J_meep includes /resolution so Meep's planar δ × gv.a cancels for per-cell ΔE.
j_unit = {j_unit!r}
c0 = {C0!r}

def my_src_func(t):
    t_sec = float(t) * 1e-3 / c0
    j_si = np.exp(-0.5 * ((t_sec - t0) / sigma) ** 2) * np.sin(2 * np.pi * freq_hz * t_sec)
    return float(j_si) * j_unit

sources = [
    mp.Source(
        mp.CustomSource(src_func=my_src_func),
        component=mp.Ex,
        center=mp.Vector3(0, 0, {z_mm!r}),
        size=mp.Vector3({src_sx!r}, {src_sy!r}, 0),
    )
]
sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell,
    boundary_layers=[mp.PML(thickness={float(npml) * dl_mm!r})],
    geometry=geometry,
    sources=sources,
    eps_averaging=False,
    force_complex_fields=False,
    Courant={float(OPENCL_COURANT)!r},
)
vol = mp.Volume(center=mp.Vector3(0, 0, 0), size=mp.Vector3({Lx}, 0, {Lz}))
{body}
print("MEEP_JSON:" + json.dumps({{"ok": True, "steps": {checkpoints!r}}}))
"""


def run_meep_planes(case: PlaneCase) -> dict[int, np.ndarray]:
    """Meep mid-plane Ex (same physical times as OpenCL checkpoints)."""
    import shutil
    import subprocess
    import sys

    from .harness import MeepUnavailableError, _docker_run, ensure_local_pymeep_image

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        script_path = work / "meep_plane_case.py"
        script_path.write_text(_meep_script(case, out_rel="."), encoding="utf-8")

        ran = False
        try:
            import meep  # noqa: F401

            subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(work),
                capture_output=True,
                text=True,
                check=True,
            )
            ran = True
        except ImportError:
            pass
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Local Meep plane script failed.\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
            ) from e

        if not ran:
            if not shutil.which("docker"):
                raise MeepUnavailableError(
                    "Meep is not installed locally and Docker is not available"
                )
            ensure_local_pymeep_image()
            try:
                _docker_run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "-v",
                        f"{work}:/work",
                        "-w",
                        "/work",
                        "local-pymeep:latest",
                        "python",
                        "meep_plane_case.py",
                    ]
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Docker Meep plane script failed.\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
                ) from e

        out: dict[int, np.ndarray] = {}
        for step in case.checkpoints:
            path = work / f"meep_ex_step{step:04d}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Meep did not write {path}")
            out[int(step)] = np.load(path)
        return out


def run_case_planes(case: PlaneCase) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    ocl = run_opencl_planes(case)
    meep = run_meep_planes(case)
    return ocl, meep


def write_case_baselines(
    case: PlaneCase,
    out_dir: str | Path,
    *,
    ocl: dict[int, np.ndarray] | None = None,
    meep: dict[int, np.ndarray] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write PNGs + float32 planes + meta.json under ``out_dir/<case.name>/``.

    Returns ``(case_dir, case_report_dict)`` for aggregate discrepancy reports.
    """
    from .plane_metrics import case_report_dict, case_summary, measure_case

    root = Path(out_dir)
    dest = root / case.name
    dest.mkdir(parents=True, exist_ok=True)
    if ocl is None or meep is None:
        ocl, meep = run_case_planes(case)
    assert ocl is not None and meep is not None
    rows = measure_case(ocl, meep, npml=case.npml, checkpoints=case.checkpoints)
    meta: dict[str, Any] = {
        "name": case.name,
        "shape": list(case.shape),
        "dl_m": case.dl,
        "npml": case.npml,
        "n_steps": case.n_steps,
        "freq_hz": case.freq,
        "fwidth_hz": case.fwidth,
        "checkpoints": list(case.checkpoints),
        "block_half": list(case.block_half),
        "block_eps": case.block_eps,
        "courant": float(OPENCL_COURANT),
        "source": "Jx",
        "rim_taper": True,
        "rim_edge": 0.8,
        "files": [],
        "discrepancy": {
            "summary": case_summary(rows),
            "checkpoints": [r.__dict__ for r in rows],
        },
    }
    for step in case.checkpoints:
        if step not in ocl or step not in meep:
            raise KeyError(f"missing checkpoint {step}")
        png = dest / f"step_{step:04d}.png"
        write_triptych_png(png, ocl[step], meep[step])
        np.save(dest / f"ocl_ex_step{step:04d}.npy", np.asarray(ocl[step], dtype=np.float32))
        np.save(dest / f"meep_ex_step{step:04d}.npy", np.asarray(meep[step], dtype=np.float32))
        meta["files"].append(png.name)
    (dest / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = case_report_dict(
        name=case.name,
        shape=list(case.shape),
        dl_m=case.dl,
        npml=case.npml,
        n_steps=case.n_steps,
        freq_hz=case.freq,
        fwidth_hz=case.fwidth,
        block_half=list(case.block_half),
        block_eps=case.block_eps,
        courant=float(OPENCL_COURANT),
        rows=rows,
        images=list(meta["files"]),
    )
    return dest, report


def load_case_planes_from_baselines(case_dir: Path, checkpoints: list[int] | tuple[int, ...]):
    """Load committed OpenCL/Meep planes for report regeneration checks."""
    ocl: dict[int, np.ndarray] = {}
    meep: dict[int, np.ndarray] = {}
    for step in checkpoints:
        ocl[int(step)] = np.load(case_dir / f"ocl_ex_step{step:04d}.npy")
        meep[int(step)] = np.load(case_dir / f"meep_ex_step{step:04d}.npy")
    return ocl, meep


def write_all_baselines(out_dir: str | Path, cases: list[PlaneCase] | None = None) -> Path:
    """Write every case plus aggregate ``DISCREPANCY_REPORT.md`` / JSON."""
    from .plane_metrics import build_discrepancy_document, write_discrepancy_reports

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if cases is None:
        cases = default_plane_cases()
    case_docs: list[dict[str, Any]] = []
    for case in cases:
        _dest, report = write_case_baselines(case, root)
        case_docs.append(report)
    doc = build_discrepancy_document(case_docs)
    write_discrepancy_reports(root, doc)
    return root
