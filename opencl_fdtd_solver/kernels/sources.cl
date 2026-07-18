/* Copyright (C) 2026: OpenCL FDTD Solver Contributors
 * Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
 *
 * This file is part of opencl-fdtd-solver (GPLv3 or later).
 */

__kernel void add_source_Ex(
    int Nx, int Ny, int Nz,
    int z_src, real amp,
    int i0, int i1, int j0, int j1,
    __global real * restrict Ex
) {
    int j = get_global_id(0);
    int i = get_global_id(1);

    if (i >= Nx || j >= Ny) return;
    if (i < i0 || i >= i1 || j < j0 || j >= j1) return;

    int idx = i * Ny * Nz + j * Nz + z_src;
    Ex[idx] += amp;
}

/* Soft current-density inject: Ex += -dt/(ε₀ εᵣ) Jx (Meep-like D -= J·dt).
 * rim_taper≠0 multiplies by sheet rim weights: edges × rim_edge, corners
 * × rim_edge² (host may renorm J so ∑w equals the hard cell count). */
__kernel void add_source_Jx(
    int Nx, int Ny, int Nz,
    int z_src, real Jx,
    int i0, int i1, int j0, int j1,
    int rim_taper,
    real rim_edge,
    __global const real * restrict ce_x,  /* dt/(eps0*eps_r) at Ex edges */
    __global real * restrict Ex
) {
    int j = get_global_id(0);
    int i = get_global_id(1);

    if (i >= Nx || j >= Ny) return;
    if (i < i0 || i >= i1 || j < j0 || j >= j1) return;

    real w = (real)1.0;
    if (rim_taper) {
        if (i == i0 || i == i1 - 1) w *= rim_edge;
        if (j == j0 || j == j1 - 1) w *= rim_edge;
    }

    int idx = i * Ny * Nz + j * Nz + z_src;
    Ex[idx] += -ce_x[idx] * Jx * w;
}
