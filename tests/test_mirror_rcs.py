# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# This file is part of opencl-fdtd-solver.

"""Absolute PEC-mirror RCS vs the large-plate physical-optics approximation.

The fixture solves for the scattered field: on the ideal, zero-thickness PEC
plate, impose E_scattered,t = -E_incident,t. Everywhere else the normal Yee +
CPML kernels advance the scattered wave. Thus no second numerical solver,
incident-field subtraction, arbitrary far-field scaling, or unresolved metal
skin depth is involved. This is a test-only PEC boundary, not a public PEC API.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np
import pyopencl as cl
from opencl_fdtd_solver import OpenCLFDTD, OpenCLNear2FarMonitor
from opencl_fdtd_solver.constants import C0, EPS0


def mirror_rcs_case(cells_per_wavelength: int):
    """Normally incident Ex plane wave on a 3λ × 3λ PEC square (SI units)."""
    wavelength = 0.06
    freq = C0 / wavelength
    p = cells_per_wavelength
    dl = wavelength / p
    shape = (9 * p, 9 * p, 6 * p)
    sim = OpenCLFDTD(shape, dl, npml=p)
    lo, hi = 3 * p, 6 * p
    z = 3 * p
    # Exact tangential PEC edge masks, not cell-centred material averaging:
    # Ex at (i+1/2,j,k), Ey at (i,j+1/2,k). Physical side length = (hi-lo)*dl.
    masks = ((slice(lo, hi), slice(lo, hi + 1), z),
             (slice(lo, hi + 1), slice(lo, hi), z))
    for axis, mask in zip(("x", "y"), masks):
        ca = np.ones(shape, dtype=sim.dtype)
        cb = np.full(shape, sim.dt / EPS0, dtype=sim.dtype)
        ca[mask] = cb[mask] = 0
        cl.enqueue_copy(sim.queue, getattr(sim, f"ca_{axis}_buf"), ca)
        cl.enqueue_copy(sim.queue, getattr(sim, f"ce_{axis}_buf"), cb)

    incident_dft = 0j
    omega = 2 * np.pi * freq

    def pec_boundary(f):
        nonlocal incident_dft
        # Analytic incident plane wave evaluated on the plate, amplitude 1 V/m.
        incident = np.exp(-0.5 * ((f.t * freq - 5.0) / 0.8) ** 2) * np.sin(omega * f.t)
        incident_dft += incident * np.exp(1j * omega * f.t) * f.dt
        # E update set the tangential PEC edges to zero; now enforce E_s=-E_i
        # at the new E time, before the N2F monitor or the next H update.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            f.add_source_Ex(z, -incident, i0=lo, i1=hi, j0=lo, j1=hi + 1)

    sim.add_monitor(pec_boundary)
    mon = OpenCLNear2FarMonitor(
        sim, (4.5 * wavelength, 4.5 * wavelength, 3 * wavelength),
        (4 * wavelength, 4 * wavelength, 2 * wavelength), freq,
    )
    # θ is bistatic observation angle from the specular (-z) direction, not
    # the tilt of the plate in a monostatic aspect-angle experiment.
    theta = np.deg2rad(np.array([0, 3, 6, 9, 12, 15, 19.4712206345]))
    distance = 10.0  # > 2*(3λ)^2/λ, safely in the Fraunhofer region
    points = distance * np.column_stack((np.sin(theta), np.zeros_like(theta), -np.cos(theta)))
    sim.run(round(20.0 / (freq * sim.dt)))
    fields = mon.get_farfields(points)
    early = fields.copy()
    sim.run(round(4.0 / (freq * sim.dt)))
    fields = mon.get_farfields(points)
    rcs = 4 * np.pi * distance**2 * np.sum(np.abs(fields[:, :3])**2, axis=1) / abs(incident_dft)**2
    side = 3 * wavelength
    # PO induced current 2*n×H_inc; E-plane projection supplies cos²θ.
    expected = (4 * np.pi * side**4 / wavelength**2
                * np.cos(theta)**2 * np.sinc(side / wavelength * np.sin(theta))**2)
    convergence = np.linalg.norm(fields - early) / np.linalg.norm(fields)
    return rcs, expected, convergence


if __name__ == "__main__":
    unittest.main()
