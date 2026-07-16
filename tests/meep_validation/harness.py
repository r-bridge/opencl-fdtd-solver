# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
#
# This file is part of opencl-fdtd-solver.

"""Shared helpers for OpenCL ↔ MEEP validation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from opencl_fdtd_solver.constants import C0, EPS0

# Match OpenCLFDTD: dt = S * dl / c with S = 0.99/sqrt(3). Meep default is 0.5.
OPENCL_COURANT = 0.99 / float(np.sqrt(3.0))
# Length unit used in Meep comparison scripts (metres per Meep length unit).
MEEP_LENGTH_M = 1e-3


def meep_jx_from_si(
    j_si: float,
    *,
    length_unit_m: float = MEEP_LENGTH_M,
    resolution: float | None = None,
    dl_meep: float | None = None,
) -> float:
    """Map SI current density so Meep ΔE matches SI ΔE numerically.

    Meep (ε₀≡1): ``ΔE = -J_meep·Δt_meep / εᵣ``.
    SI: ``ΔE = -J_si·Δt_si / (ε₀ εᵣ)`` with ``Δt_si = Δt_meep·(length_unit_m/c)``.

    Therefore raw ``J_meep = J_si · length_unit_m / (ε₀ c)``.

    For a **planar** (zero-thickness) volume source Meep additionally multiplies
    the stored amplitude by ``gv.a`` (= ``resolution``) so the δ-function current
    retains resolution-independent ∫J. Pass ``resolution`` (or ``dl_meep =
    1/resolution``) to cancel that factor when matching per-cell ΔE to SI Jx on
    a Yee sheet.
    """
    if resolution is not None and dl_meep is not None:
        raise ValueError("pass only one of resolution or dl_meep")
    j = float(j_si) * float(length_unit_m) / (float(EPS0) * C0)
    if resolution is not None:
        j /= float(resolution)
    elif dl_meep is not None:
        j *= float(dl_meep)
    return j


class MeepUnavailableError(RuntimeError):
    """Raised when neither local Meep nor Docker Meep can be executed."""


def opencl_dt(dl: float) -> float:
    return float(OPENCL_COURANT) * float(dl) / C0


def meep_until(n_steps: int, dl: float) -> float:
    """Physical OpenCL runtime in Meep units with length unit = 1 mm."""
    return float(n_steps) * float(OPENCL_COURANT) * (float(dl) / 1e-3)


def gaussian_sine_amp(t: float, freq: float, fwidth: float) -> float:
    t0 = 5.0 / (np.pi * fwidth)
    sigma = 1.0 / (np.pi * fwidth)
    return float(np.exp(-0.5 * ((t - t0) / sigma) ** 2) * np.sin(2 * np.pi * freq * t))


def peak_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(x)))
    if peak <= 0.0:
        raise ValueError("cannot peak-normalize an all-zero array")
    return x / peak


def max_abs_db_error(a_db: np.ndarray, b_db: np.ndarray, *, mask_db: float | None = None) -> float:
    """Max |Δ| after aligning peaks to 0 dB.

    If ``mask_db`` is set (e.g. -15), only angles where *both* patterns are within
    that many dB of their own peak are compared (ignores deep-null floor mismatch).
    """
    a = np.asarray(a_db, dtype=np.float64)
    b = np.asarray(b_db, dtype=np.float64)
    a0 = a - np.max(a)
    b0 = b - np.max(b)
    if mask_db is None:
        return float(np.max(np.abs(a0 - b0)))
    mask = (a0 >= mask_db) & (b0 >= mask_db)
    if not np.any(mask):
        return float(np.max(np.abs(a0 - b0)))
    return float(np.max(np.abs(a0[mask] - b0[mask])))


def rel_mag_error(a: complex | float, b: complex | float) -> float:
    aa = abs(complex(a))
    bb = abs(complex(b))
    return float(abs(aa - bb) / max(bb, 1e-30))


def complex_align(a: complex, b: complex) -> complex:
    """Multiply a by e^{jφ} so arg(a') matches arg(b)."""
    if abs(a) < 1e-30 or abs(b) < 1e-30:
        return complex(a)
    return complex(a * np.exp(1j * (np.angle(b) - np.angle(a))))


def ensure_pyopencl_ctx() -> None:
    os.environ["PYOPENCL_CTX"] = os.environ.get("PYOPENCL_CTX", "0")


def _docker_run(args: list[str]) -> subprocess.CompletedProcess:
    use_sg = sys.platform.startswith("linux")
    if use_sg:
        return subprocess.run(
            ["sg", "docker", "-c", " ".join(args)],
            capture_output=True,
            text=True,
            check=True,
        )
    return subprocess.run(args, capture_output=True, text=True, check=True)


def ensure_local_pymeep_image() -> None:
    if not shutil.which("docker"):
        raise MeepUnavailableError("Docker is not available")
    try:
        _docker_run(["docker", "image", "inspect", "local-pymeep:latest"])
        return
    except subprocess.CalledProcessError:
        pass

    dockerfile = """FROM continuumio/miniconda3:latest
RUN conda create -n pymeep-env -c conda-forge python=3.11 pymeep -y
ENV PATH /opt/conda/envs/pymeep-env/bin:$PATH
"""
    build_dir = tempfile.mkdtemp()
    try:
        with open(os.path.join(build_dir, "Dockerfile"), "w", encoding="utf-8") as df:
            df.write(dockerfile)
        _docker_run(["docker", "build", "-t", "local-pymeep:latest", build_dir])
    except subprocess.CalledProcessError as e:
        raise MeepUnavailableError(
            f"Failed to build local-pymeep image.\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
        ) from e
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)


def run_meep_script(script: str) -> str:
    """Run a Meep script locally or via Docker; return stdout."""
    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = os.path.join(temp_dir, "meep_case.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)

        try:
            import meep  # noqa: F401

            res = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                check=True,
            )
            return res.stdout
        except ImportError:
            pass
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Local Meep script failed.\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
            ) from e

        if not shutil.which("docker"):
            raise MeepUnavailableError("Meep is not installed locally and Docker is not available")
        ensure_local_pymeep_image()
        try:
            res = _docker_run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{temp_dir}:/work",
                    "-w",
                    "/work",
                    "local-pymeep:latest",
                    "python",
                    "meep_case.py",
                ]
            )
            return res.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Docker Meep script failed.\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
            ) from e


def parse_meep_json(stdout: str, key: str = "MEEP_JSON") -> Any:
    for line in stdout.splitlines():
        if line.startswith(key + ":"):
            payload = line.split(":", 1)[1].strip()
            return json.loads(payload)
    raise ValueError(f"No {key}: line found in Meep stdout:\n{stdout}")


def poynting_db_from_eh(ff: np.ndarray) -> tuple[float, float]:
    E = ff[0:3]
    H = ff[3:6]
    Sx = 0.5 * (E[1] * np.conj(H[2]) - E[2] * np.conj(H[1]))
    Sy = 0.5 * (E[2] * np.conj(H[0]) - E[0] * np.conj(H[2]))
    Sz = 0.5 * (E[0] * np.conj(H[1]) - E[1] * np.conj(H[0]))
    mag = float(np.sqrt(np.abs(Sx) ** 2 + np.abs(Sy) ** 2 + np.abs(Sz) ** 2))
    db = float(20.0 * np.log10(max(mag, 1e-30)))
    return db, mag
