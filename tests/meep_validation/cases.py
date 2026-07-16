# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
#
# This file is part of opencl-fdtd-solver.

"""Case runners: OpenCL side + matching Meep scripts."""

from __future__ import annotations

from typing import Any

import numpy as np
from opencl_fdtd_solver import OpenCLFDTD, OpenCLNear2FarMonitor

from . import (
    ensure_pyopencl_ctx,
    gaussian_sine_amp,
    meep_until,
    parse_meep_json,
    run_meep_script,
)

# Shared reference geometry (matches historical compare_with_meep.py)
SHAPE = (30, 30, 30)
DL = 2e-3
NPML = 6
FREQ = 5e9
FWIDTH = 0.2 * FREQ
N_STEPS = 500
MONITOR_CTR = (30e-3, 30e-3, 30e-3)
MONITOR_SIZE = (20e-3, 20e-3, 20e-3)
# SI |S| at 1000 m underflows float32 noise on this tiny grid; angular shape is
# distance-independent in the far-field limit, so compare at 10 m for OpenCL.
OBS_R_OPENCL = 10.0  # metres
OBS_R_MEEP_MM = 1e6  # 1000 m in Meep mm units (historical / well in far field)


def _z_src() -> int:
    """Legacy full-sheet source plane (used only where noted)."""
    return SHAPE[2] - NPML - 2


def _center_src_plane() -> tuple[int, int, int, int, int]:
    """Compact Ex patch at domain center, strictly inside the Huygens box."""
    ci, cj, ck = SHAPE[0] // 2, SHAPE[1] // 2, SHAPE[2] // 2
    half = 2  # 5×5 patch
    return ci - half, ci + half, cj - half, cj + half, ck


def _make_solver(eps: np.ndarray | None = None, *, compact_source: bool = False) -> OpenCLFDTD:
    ensure_pyopencl_ctx()
    fdtd = OpenCLFDTD(SHAPE, DL, npml=NPML)
    if eps is not None:
        fdtd.set_epsilon(eps)

    if compact_source:
        i0, i1, j0, j1, k_src = _center_src_plane()

        def source_cb(f):
            amp = gaussian_sine_amp(f.t, FREQ, FWIDTH)
            ex = f.Ex
            ex[i0 : i1 + 1, j0 : j1 + 1, k_src] += amp
            import pyopencl as cl

            cl.enqueue_copy(f.queue, f.Ex_buf, np.ascontiguousarray(ex))

        fdtd.add_source(source_cb)
    else:
        z = _z_src()

        def source_cb(f):
            f.add_source_Ex(z, gaussian_sine_amp(f.t, FREQ, FWIDTH))

        fdtd.add_source(source_cb)
    return fdtd


def _sphere_eps(eps_r: float = 4.0, rad_cells: int = 6) -> np.ndarray:
    eps = np.ones(SHAPE, dtype=np.float32)
    ctr = SHAPE[0] // 2
    for i in range(SHAPE[0]):
        for j in range(SHAPE[1]):
            for k in range(SHAPE[2]):
                if (i - ctr) ** 2 + (j - ctr) ** 2 + (k - ctr) ** 2 < rad_cells**2:
                    eps[i, j, k] = eps_r
    return eps


def _meep_common_preamble(
    eps_sphere: float | None = None,
    *,
    compact_source: bool = True,
) -> str:
    """Meep script fragment: geometry, source, optional dielectric sphere."""
    until = meep_until(N_STEPS, DL)
    sphere_block = ""
    if eps_sphere is not None:
        sphere_block = f"""
geometry = [
    mp.Sphere(radius=12.0, center=mp.Vector3(), material=mp.Medium(epsilon={eps_sphere}))
]
"""
    else:
        sphere_block = "geometry = []\n"

    if compact_source:
        # 5 cells × 2 mm = 10 mm patch at center
        src_block = """
sources = [
    mp.Source(
        mp.CustomSource(src_func=my_src_func),
        component=mp.Ex,
        center=mp.Vector3(0, 0, 0),
        size=mp.Vector3(10, 10, 0),
    )
]
"""
    else:
        z_src_mm = (_z_src() - SHAPE[2] / 2.0) * (DL * 1e3)
        src_block = f"""
sources = [
    mp.Source(
        mp.CustomSource(src_func=my_src_func),
        component=mp.Ex,
        center=mp.Vector3(0, 0, {z_src_mm}),
        size=mp.Vector3(60, 60, 0),
    )
]
"""

    return f"""
import meep as mp
import numpy as np
import json

resolution = 0.5  # pixels / mm  → 30 cells over 60 mm
cell = mp.Vector3(60, 60, 60)
boundary_layers = [mp.PML(thickness=12.0)]
{sphere_block}
freq_hz = {FREQ}
fwidth_hz = {FWIDTH}
t0 = 5.0 / (np.pi * fwidth_hz)
sigma = 1.0 / (np.pi * fwidth_hz)

def my_src_func(t):
    t_sec = t * 1e-3 / 299792458.0
    return np.exp(-0.5 * ((t_sec - t0) / sigma)**2) * np.sin(2 * np.pi * freq_hz * t_sec)

{src_block}

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell,
    boundary_layers=boundary_layers,
    sources=sources,
    geometry=geometry,
    eps_averaging=False,
)

f_meep = 1.0 / 60.0  # 5 GHz with a=1 mm
UNTIL = {until}
"""


def run_opencl_nearfield_dft(probe_ijk: list[tuple[int, int, int]]) -> dict[str, Any]:
    fdtd = _make_solver(compact_source=False)
    omega = 2.0 * np.pi * FREQ
    dfts = {p: 0j for p in probe_ijk}
    for _ in range(N_STEPS):
        fdtd.step()
        phase = np.exp(1j * omega * fdtd.t) * fdtd.dt
        for i, j, k in probe_ijk:
            dfts[(i, j, k)] += fdtd.read_point("Ex", i, j, k) * phase
    return {
        "probes": [
            {
                "ijk": [i, j, k],
                "Ex_dft_real": complex(dfts[(i, j, k)]).real,
                "Ex_dft_imag": complex(dfts[(i, j, k)]).imag,
            }
            for i, j, k in probe_ijk
        ]
    }


def run_meep_nearfield_dft(probe_ijk: list[tuple[int, int, int]]) -> dict[str, Any]:
    # OpenCL index i sits at x=i*dl; Meep origin is domain center.
    probes_mm = []
    for i, j, k in probe_ijk:
        x = (i - SHAPE[0] / 2.0) * (DL * 1e3)
        y = (j - SHAPE[1] / 2.0) * (DL * 1e3)
        z = (k - SHAPE[2] / 2.0) * (DL * 1e3)
        probes_mm.append((x, y, z))

    probe_list = ",\n".join(f"    mp.Vector3({x}, {y}, {z})" for x, y, z in probes_mm)
    script = (
        _meep_common_preamble(None, compact_source=False)
        + f"""
probe_pts = [
{probe_list}
]
omega = 2 * np.pi * freq_hz
c0 = 299792458.0
series = [[] for _ in probe_pts]
times = []

def record(sim):
    times.append(sim.meep_time())
    for i, pt in enumerate(probe_pts):
        series[i].append(complex(sim.get_field_point(mp.Ex, pt)))

sim.run(mp.at_every(0.5, record), until=UNTIL)

out = []
for pt, ser in zip(probe_pts, series):
    acc = 0j
    for n in range(len(ser)):
        t_meep = times[n]
        t_sec = t_meep * 1e-3 / c0
        if n + 1 < len(times):
            dt_sec = (times[n + 1] - times[n]) * 1e-3 / c0
        elif n > 0:
            dt_sec = (times[n] - times[n - 1]) * 1e-3 / c0
        else:
            dt_sec = 0.5 * 1e-3 / c0
        acc += ser[n] * np.exp(1j * omega * t_sec) * dt_sec
    out.append({{"x_mm": pt.x, "y_mm": pt.y, "z_mm": pt.z,
                "Ex_dft_real": complex(acc).real, "Ex_dft_imag": complex(acc).imag}})
print("MEEP_JSON:" + json.dumps({{"probes": out}}))
"""
    )
    return parse_meep_json(run_meep_script(script))


def run_opencl_farfield_pattern(
    n_angles: int = 19, eps: np.ndarray | None = None
) -> dict[str, Any]:
    fdtd = _make_solver(eps, compact_source=True)
    mon = OpenCLNear2FarMonitor(fdtd, MONITOR_CTR, MONITOR_SIZE, FREQ)
    fdtd.run(N_STEPS)
    R = OBS_R_OPENCL
    angles, db = mon.farfield_polar_xz(distance_m=R, n_angles=n_angles)
    eh_z = mon.get_farfield((0.0, 0.0, R))
    eh_x = mon.get_farfield((R, 0.0, 0.0))
    return {
        "angles_deg": angles.tolist(),
        "S_db": db.tolist(),
        "eh_plus_z": _eh_to_list(eh_z),
        "eh_plus_x": _eh_to_list(eh_x),
        "obs_r_m": R,
    }


def run_meep_farfield_pattern(
    n_angles: int = 19, eps_sphere: float | None = None
) -> dict[str, Any]:
    script = (
        _meep_common_preamble(eps_sphere, compact_source=True)
        + f"""
R = 5.0
n2f_regions = [
    mp.Near2FarRegion(center=mp.Vector3(x=-2*R), size=mp.Vector3(0, 4*R, 4*R), weight=+1),
    mp.Near2FarRegion(center=mp.Vector3(x=+2*R), size=mp.Vector3(0, 4*R, 4*R), weight=-1),
    mp.Near2FarRegion(center=mp.Vector3(y=-2*R), size=mp.Vector3(4*R, 0, 4*R), weight=+1),
    mp.Near2FarRegion(center=mp.Vector3(y=+2*R), size=mp.Vector3(4*R, 0, 4*R), weight=-1),
    mp.Near2FarRegion(center=mp.Vector3(z=-2*R), size=mp.Vector3(4*R, 4*R, 0), weight=+1),
    mp.Near2FarRegion(center=mp.Vector3(z=+2*R), size=mp.Vector3(4*R, 4*R, 0), weight=-1),
]
n2f = sim.add_near2far(f_meep, 0, 1, *n2f_regions)
sim.run(until=UNTIL)

n_angles = {n_angles}
angles = np.linspace(-180.0, 180.0, n_angles)
S_db = []
def eh_list(ff):
    return [complex(v).real for v in ff] + [complex(v).imag for v in ff]  # placeholder replaced below

def pack_eh(ff):
    out = []
    for v in ff:
        c = complex(v)
        out.append({{"re": c.real, "im": c.imag}})
    return out

Rmm = {OBS_R_MEEP_MM}
for ang in angles:
    rad = np.deg2rad(ang)
    obs = mp.Vector3(Rmm * np.sin(rad), 0, Rmm * np.cos(rad))
    ff = sim.get_farfield(n2f, obs)
    E = np.array(ff[0:3]); H = np.array(ff[3:6])
    Sx = 0.5 * (E[1]*np.conj(H[2]) - E[2]*np.conj(H[1]))
    Sy = 0.5 * (E[2]*np.conj(H[0]) - E[0]*np.conj(H[2]))
    Sz = 0.5 * (E[0]*np.conj(H[1]) - E[1]*np.conj(H[0]))
    mag = float(np.sqrt(np.abs(Sx)**2 + np.abs(Sy)**2 + np.abs(Sz)**2))
    S_db.append(20.0 * np.log10(max(mag, 1e-30)))

ff_z = sim.get_farfield(n2f, mp.Vector3(0, 0, Rmm))
ff_x = sim.get_farfield(n2f, mp.Vector3(Rmm, 0, 0))
print("MEEP_JSON:" + json.dumps({{
    "angles_deg": angles.tolist(),
    "S_db": S_db,
    "eh_plus_z": pack_eh(ff_z),
    "eh_plus_x": pack_eh(ff_x),
}}))
"""
    )
    return parse_meep_json(run_meep_script(script))


def run_opencl_pml_decay() -> dict[str, Any]:
    """Pulse then free decay: peak Ex energy vs energy after long quiet run."""
    fdtd = _make_solver(compact_source=False)
    e_peak = 0.0
    peak_step = 0
    n_drive = N_STEPS
    n_total = N_STEPS * 4
    e_late = 0.0
    for step in range(n_total):
        if step == n_drive:
            fdtd.clear_sources()
        fdtd.step()
        if step % 25 == 0 or step == n_total - 1:
            ex = fdtd.Ex
            e = float(np.sum(ex * ex))
            if e > e_peak:
                e_peak = e
                peak_step = step
            if step == n_total - 1:
                e_late = e
    return {
        "energy_peak": e_peak,
        "energy_late": e_late,
        "peak_step": peak_step,
        "ratio_late_over_peak": e_late / max(e_peak, 1e-30),
    }


def run_meep_pml_decay() -> dict[str, Any]:
    until_drive = meep_until(N_STEPS, DL)
    until_total = meep_until(N_STEPS * 4, DL)
    script = (
        _meep_common_preamble(None, compact_source=False)
        + f"""
def ex_energy(sim):
    ex = sim.get_array(component=mp.Ex, center=mp.Vector3(), size=cell)
    return float(np.sum(np.abs(ex)**2))

energies = []
def track(sim):
    energies.append(ex_energy(sim))

sim.run(mp.at_every(5.0, track), until={until_drive})
sim.run(mp.at_every(5.0, track), until={until_total} - {until_drive})
e_peak = float(max(energies)) if energies else 0.0
e_late = float(energies[-1]) if energies else 0.0
print("MEEP_JSON:" + json.dumps({{
    "energy_peak": e_peak,
    "energy_late": e_late,
    "ratio_late_over_peak": e_late / max(e_peak, 1e-30),
}}))
"""
    )
    return parse_meep_json(run_meep_script(script))


def _eh_to_list(ff: np.ndarray) -> list[dict[str, float]]:
    out = []
    for v in ff:
        c = complex(v)
        out.append({"re": c.real, "im": c.imag})
    return out


def eh_from_list(items: list[dict[str, float]]) -> np.ndarray:
    return np.array([complex(d["re"], d["im"]) for d in items], dtype=np.complex128)
