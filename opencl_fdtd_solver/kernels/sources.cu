/* Copyright (C) 2026: OpenCL FDTD Solver Contributors
 * Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
 *
 * This file is part of opencl-fdtd-solver (GPLv3 or later).
 *
 * CUDA port of sources.cl (see yee_update.cu for the real/real2 defines).
 */

extern "C" __global__ void add_source_Ex(
    int Nx, int Ny, int Nz,
    int z_src, real amp,
    int i0, int i1, int j0, int j1,
    real * __restrict__ Ex
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= Nx || j >= Ny) return;
    if (i < i0 || i >= i1 || j < j0 || j >= j1) return;

    int idx = i * Ny * Nz + j * Nz + z_src;
    Ex[idx] += amp;
}

/* Soft current-density inject: Ex += -dt/(eps0*eps_r) Jx (Meep-like D -= J.dt).
 * rim_taper!=0 multiplies by sheet rim weights: edges x rim_edge, corners
 * x rim_edge^2 (host may renorm J so sum w equals the hard cell count). */
extern "C" __global__ void add_source_Jx(
    int Nx, int Ny, int Nz,
    int z_src, real Jx,
    int i0, int i1, int j0, int j1,
    int rim_taper,
    real rim_edge,
    const real * __restrict__ ce_x,  /* dt/(eps0*eps_r) at Ex edges */
    real * __restrict__ Ex
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= Nx || j >= Ny) return;
    if (i < i0 || i >= i1 || j < j0 || j >= j1) return;

    real w = (real)1;
    if (rim_taper) {
        if (i == i0 || i == i1 - 1) w *= rim_edge;
        if (j == j0 || j == j1 - 1) w *= rim_edge;
    }

    int idx = i * Ny * Nz + j * Nz + z_src;
    Ex[idx] += -ce_x[idx] * Jx * w;
}
