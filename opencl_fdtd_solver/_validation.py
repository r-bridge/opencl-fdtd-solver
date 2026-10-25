"""Shared grid and sheet-source validation (internal)."""

import operator

import numpy as np

COURANT_SAFETY_FACTOR = 0.99
MIN_EPSILON_R = COURANT_SAFETY_FACTOR**2


def _index(value, name):
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def validate_grid(shape, dl, npml):
    shape = tuple(_index(v, "shape entries") for v in shape)
    if len(shape) != 3 or any(v <= 0 for v in shape):
        raise ValueError("shape must contain three positive integers")
    dl = float(dl)
    if not np.isfinite(dl) or dl <= 0:
        raise ValueError("dl must be finite and positive")
    npml = _index(npml, "npml")
    if npml < 0 or 2 * npml > min(shape):
        raise ValueError("npml must be nonnegative and 2*npml must not exceed any grid dimension")
    return shape, dl, npml


def validate_material_arrays(eps_array, sigma_array, expected_shape):
    """Return real material arrays after validating shape, values, and CFL bound."""

    def _real_array(values, name):
        if np.iscomplexobj(values):
            raise ValueError(f"{name} must contain real values")
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain real numeric values") from exc
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} shape mismatch: expected {expected_shape}, got {tuple(array.shape)}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array

    eps = _real_array(eps_array, "Epsilon")
    if np.any(eps < MIN_EPSILON_R):
        raise ValueError(
            f"Relative permittivity must be at least {MIN_EPSILON_R:g} "
            f"for the fixed Courant factor {COURANT_SAFETY_FACTOR:g}"
        )

    sigma = None
    if sigma_array is not None:
        sigma = _real_array(sigma_array, "Sigma")
        if np.any(sigma < 0):
            raise ValueError("Conductivity must be nonnegative")
    return eps, sigma


def source_sheet(shape, z, i0, i1, j0, j1):
    nx, ny, nz = shape
    z = _index(z, "z_src")
    i0 = 0 if i0 is None else _index(i0, "i0")
    i1 = nx if i1 is None else _index(i1, "i1")
    j0 = 0 if j0 is None else _index(j0, "j0")
    j1 = ny if j1 is None else _index(j1, "j1")
    if not 0 <= z < nz:
        raise ValueError("z_src must satisfy 0 <= z_src < Nz")
    if not (0 <= i0 <= i1 <= nx and 0 <= j0 <= j1 <= ny):
        raise ValueError("sheet bounds must satisfy 0 <= lo <= hi <= the grid dimension")
    return z, i0, i1, j0, j1


def sheet_current(jx, nx, ny, rim_taper, rim_edge, rim_renorm):
    jx, re = float(jx), float(rim_edge)
    if not np.isfinite(jx):
        raise ValueError("Jx must be finite")
    if not np.isfinite(re) or re < 0:
        raise ValueError("rim_edge must be finite and nonnegative")
    if rim_taper and rim_renorm and nx and ny:
        # A one-cell span is an edge once, as in the OpenCL kernel's OR test.
        wx = re if nx == 1 else nx - 2 + 2 * re
        wy = re if ny == 1 else ny - 2 + 2 * re
        wsum = wx * wy
        if wsum <= 0:
            raise ValueError("cannot renormalize a sheet whose rim weights are all zero")
        jx *= nx * ny / wsum
    return jx, re
