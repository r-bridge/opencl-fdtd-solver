# Copyright (C) 2026: OpenCL FDTD Solver Contributors
#
# This file is part of opencl-fdtd-solver.

"""Mie scattering amplitudes for a non-magnetic dielectric sphere (Bohren–Huffman)."""

from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn, spherical_yn


def _riccati_psi(n: int, z: complex | float) -> complex:
    """ψ_n(z) = z j_n(z)."""
    zc = complex(z)
    return zc * complex(spherical_jn(n, zc))


def _riccati_xi(n: int, z: complex | float) -> complex:
    """ξ_n(z) = z (j_n(z) + i y_n(z))."""
    zc = complex(z)
    return zc * (complex(spherical_jn(n, zc)) + 1j * complex(spherical_yn(n, zc)))


def _riccati_psi_deriv(n: int, z: complex | float) -> complex:
    """dψ_n/dz: ψ'_n = −ψ_{n+1} + (n+1) ψ_n / z  (stable upward form)."""
    zc = complex(z)
    return (n + 1) * _riccati_psi(n, zc) / zc - _riccati_psi(n + 1, zc)


def _riccati_xi_deriv(n: int, z: complex | float) -> complex:
    zc = complex(z)
    return (n + 1) * _riccati_xi(n, zc) / zc - _riccati_xi(n + 1, zc)


def mie_nstop(x: float) -> int:
    """Wiscombe truncation for size parameter ``x = ka``."""
    return int(x + 4.0 * x ** (1.0 / 3.0) + 2.0) + 2


def mie_coefficients(
    m: complex | float, x: float, nstop: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Electric (a_n) and magnetic (b_n) Mie coefficients for μ=1 sphere."""
    if nstop is None:
        nstop = mie_nstop(float(x))
    m = complex(m)
    x = float(x)
    mx = m * x
    a = np.zeros(nstop, dtype=np.complex128)
    b = np.zeros(nstop, dtype=np.complex128)
    for n in range(1, nstop + 1):
        psi_x = _riccati_psi(n, x)
        xi_x = _riccati_xi(n, x)
        dpsi_x = _riccati_psi_deriv(n, x)
        dxi_x = _riccati_xi_deriv(n, x)
        psi_mx = _riccati_psi(n, mx)
        dpsi_mx = _riccati_psi_deriv(n, mx)
        denom_a = m * psi_mx * dxi_x - xi_x * dpsi_mx
        denom_b = psi_mx * dxi_x - m * xi_x * dpsi_mx
        a[n - 1] = (m * psi_mx * dpsi_x - psi_x * dpsi_mx) / denom_a
        b[n - 1] = (psi_mx * dpsi_x - m * psi_x * dpsi_mx) / denom_b
    return a, b


def mie_S1_S2(
    m: complex | float,
    x: float,
    theta: np.ndarray,
    nstop: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scattering amplitudes S₁(θ), S₂(θ) (θ = angle from +z, radians)."""
    a, b = mie_coefficients(m, x, nstop=nstop)
    nmax = len(a)
    mu = np.cos(np.asarray(theta, dtype=np.float64))
    # π_0 = 0, π_1 = 1; τ_n = n μ π_n − (n+1) π_{n−1}
    pi_nm2 = np.zeros_like(mu)
    pi_nm1 = np.zeros_like(mu)
    S1 = np.zeros_like(mu, dtype=np.complex128)
    S2 = np.zeros_like(mu, dtype=np.complex128)
    for n in range(1, nmax + 1):
        if n == 1:
            pi_n = np.ones_like(mu)
        else:
            pi_n = ((2 * n - 1) * mu * pi_nm1 - n * pi_nm2) / (n - 1)
        tau_n = n * mu * pi_n - (n + 1) * pi_nm1
        fn = (2 * n + 1) / (n * (n + 1))
        S1 += fn * (a[n - 1] * pi_n + b[n - 1] * tau_n)
        S2 += fn * (a[n - 1] * tau_n + b[n - 1] * pi_n)
        pi_nm2, pi_nm1 = pi_nm1, pi_n
    return S1, S2


def mie_eplane_intensity(m: complex | float, x: float, theta: np.ndarray) -> np.ndarray:
    """E-plane intensity ∝ |S₂(θ)|² for x-polarized incidence along +z."""
    _s1, s2 = mie_S1_S2(m, x, theta)
    return np.abs(s2) ** 2
