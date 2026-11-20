"""Shared grid and sheet-source validation (internal)."""

import operator

import numpy as np

COURANT_SAFETY_FACTOR = 0.99
# The fixed timestep dt = 0.99*dl/(c0*sqrt(3)) stays inside the 3-D CFL limit
# for any eps_r >= 0.99**2 = 0.9801. The accepted range is nevertheless
# eps_r >= 1: this is a nondispersive solver, where eps_r < 1 has no physical
# meaning, and 1.0 (unlike 0.9801) is exactly representable in float32, so the
# bound is inclusive for the float32 arrays callers normally pass.
MIN_EPSILON_R = 1.0


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


def _reject_biaxial_in_pml(eps, npml):
    """Raise if a biaxial cell lies in the PML normal to its intermediate-ε axis.

    A medium with three distinct diagonal components supports waves whose
    phase and group velocities have opposite signs along the axis of the
    intermediate component. A PML normal to that axis amplifies those waves
    instead of absorbing them, and the run grows without bound. PMLs normal
    to the axes of the largest and smallest components are unaffected, as
    are isotropic and uniaxial media.
    """
    for axis, name in enumerate("xyz"):
        normal = eps[axis]
        other_a, other_b = (eps[a] for a in range(3) if a != axis)
        n = normal.shape[axis]
        for slab in (slice(0, npml), slice(n - npml, n)):
            index = tuple(slab if a == axis else slice(None) for a in range(3))
            eps_n = np.asarray(normal[index], dtype=np.float64)
            diff_a = np.asarray(other_a[index], dtype=np.float64) - eps_n
            diff_b = np.asarray(other_b[index], dtype=np.float64) - eps_n
            # Opposite signs: eps_n lies strictly between the other two.
            if np.any(diff_a * diff_b < 0):
                raise ValueError(
                    f"Biaxial permittivity reaches the {name}-normal PML: cells there have "
                    f"eps_{name} strictly between the other two components, and the CPML is "
                    "unstable for such a medium (the fields grow without bound). Inside "
                    f"that PML make eps_{name} equal to one of the other two components "
                    "(isotropic or uniaxial padding), or run with npml=0"
                )


def validate_material_arrays(eps_array, sigma_array, expected_shape, npml=0):
    """Return ``(eps, sigma)`` after validating shape, values, and the εᵣ bound.

    ``eps_array`` may be a single 3-D isotropic array of shape ``expected_shape``
    or a 3-tuple ``(eps_x, eps_y, eps_z)`` for diagonal anisotropic media.
    Each array is checked on a float64 working copy (realness, finiteness,
    ``eps >= MIN_EPSILON_R``, ``sigma >= 0``). Inputs that are already real
    numeric NumPy arrays (bool, integer or float dtype) are returned as the
    caller's own object, so no second copy of a large grid is kept alive;
    anything else (nested lists, object arrays) comes back as the float64
    copy. Callers convert to their working dtype themselves.

    With ``npml > 0`` an anisotropic tuple is also rejected where a biaxial
    cell lies in the PML normal to its intermediate-permittivity axis (see
    ``_reject_biaxial_in_pml``).
    """

    def _real_array(values, name):
        try:
            original = np.asarray(values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain real numeric values") from exc
        if np.iscomplexobj(original):
            raise ValueError(f"{name} must contain real values")
        try:
            array = np.asarray(original, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain real numeric values") from exc
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} shape mismatch: expected {expected_shape}, got {tuple(array.shape)}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return original if original.dtype.kind in "biuf" else array

    def _check_eps_one(values, name):
        arr = _real_array(values, name)
        if np.any(arr < MIN_EPSILON_R):
            raise ValueError(
                f"Relative permittivity must be at least {MIN_EPSILON_R:g} "
                "(nondispersive dielectric; values below 1 would also exceed the "
                f"fixed Courant factor {COURANT_SAFETY_FACTOR:g} below "
                f"{COURANT_SAFETY_FACTOR**2:g})"
            )
        return arr

    def _has_shape(value, shape):
        try:
            return np.shape(value) == shape
        except ValueError:
            return False

    # A tuple is either an isotropic grid given as Nx planes or the three
    # components of a diagonal anisotropic medium. Anything else (a wrong
    # plane shape, too few planes) is reported as the shape error it is.
    is_tuple = isinstance(eps_array, tuple)
    isotropic_tuple = (
        is_tuple
        and len(eps_array) == expected_shape[0]
        and all(_has_shape(plane, expected_shape[1:]) for plane in eps_array)
    )
    anisotropic = is_tuple and not isotropic_tuple and len(eps_array) == 3
    if (
        is_tuple
        and not isotropic_tuple
        and not anisotropic
        and eps_array
        and all(_has_shape(comp, expected_shape) for comp in eps_array)
    ):
        raise ValueError(
            f"Anisotropic epsilon tuple must have 3 components (eps_x, eps_y, eps_z); got {len(eps_array)}"
        )
    if anisotropic:
        eps = tuple(
            _check_eps_one(comp, f"Epsilon[{ax}]") for ax, comp in zip(("x", "y", "z"), eps_array)
        )
        if npml > 0:
            _reject_biaxial_in_pml(eps, npml)
    else:
        eps = _check_eps_one(eps_array, "Epsilon")

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
