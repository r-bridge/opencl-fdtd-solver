/* Copyright (C) 2026: OpenCL FDTD Solver Contributors
 * Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
 *
 * This file is part of opencl-fdtd-solver (GPLv3 or later).
 */

/* Work-item mapping: get_global_id(0)=k, (1)=j, (2)=i so adjacent threads
 * touch contiguous addresses along the fastest array axis (k).
 */

__kernel void update_H_interior(
    int Nx, int Ny, int Nz,
    int npml,
    real dtm_dl,  /* dt / (mu0 * dl), folded on host */
    __global const real * restrict Ex,
    __global const real * restrict Ey,
    __global const real * restrict Ez,
    __global real * restrict Hx,
    __global real * restrict Hy,
    __global real * restrict Hz
) {
    int k = get_global_id(0) + npml;
    int j = get_global_id(1) + npml;
    int i = get_global_id(2) + npml;

    if (i >= Nx - npml || j >= Ny - npml || k >= Nz - npml) return;

    int idx = i * Ny * Nz + j * Nz + k;

    real dEz_dy = Ez[idx + Nz] - Ez[idx];
    real dEy_dz = Ey[idx + 1] - Ey[idx];
    real dEx_dz = Ex[idx + 1] - Ex[idx];
    real dEz_dx = Ez[idx + Ny * Nz] - Ez[idx];
    real dEy_dx = Ey[idx + Ny * Nz] - Ey[idx];
    real dEx_dy = Ex[idx + Nz] - Ex[idx];

    Hx[idx] -= dtm_dl * (dEz_dy - dEy_dz);
    Hy[idx] -= dtm_dl * (dEx_dz - dEz_dx);
    Hz[idx] -= dtm_dl * (dEy_dx - dEx_dy);
}

__kernel void update_H_pml(
    int Nx, int Ny, int Nz,
    int npml,
    real dtm,
    __global const real * restrict Ex,
    __global const real * restrict Ey,
    __global const real * restrict Ez,
    __global real * restrict Hx,
    __global real * restrict Hy,
    __global real * restrict Hz,
    /* ikx/iky/ikz hold 1/(kappa * dl), precomputed on host */
    __global const real * restrict bx, __global const real * restrict cx, __global const real * restrict ikx,
    __global const real * restrict by, __global const real * restrict cy, __global const real * restrict iky,
    __global const real * restrict bz, __global const real * restrict cz, __global const real * restrict ikz,
    __global real * restrict psi_Hx_y, __global real * restrict psi_Hx_z,
    __global real * restrict psi_Hy_x, __global real * restrict psi_Hy_z,
    __global real * restrict psi_Hz_x, __global real * restrict psi_Hz_y
) {
    int k = get_global_id(0);
    int j = get_global_id(1);
    int i = get_global_id(2);

    if (i >= Nx || j >= Ny || k >= Nz) return;

    if (npml > 0 &&
        i >= npml && i < Nx - npml &&
        j >= npml && j < Ny - npml &&
        k >= npml && k < Nz - npml) {
        return;
    }

    int idx = i * Ny * Nz + j * Nz + k;

    real dEz_dy = (j < Ny - 1) ? (Ez[idx + Nz] - Ez[idx]) : (real)0.0;
    real dEy_dz = (k < Nz - 1) ? (Ey[idx + 1] - Ey[idx])  : (real)0.0;
    real dEx_dz = (k < Nz - 1) ? (Ex[idx + 1] - Ex[idx])  : (real)0.0;
    real dEz_dx = (i < Nx - 1) ? (Ez[idx + Ny * Nz] - Ez[idx]) : (real)0.0;
    real dEy_dx = (i < Nx - 1) ? (Ey[idx + Ny * Nz] - Ey[idx]) : (real)0.0;
    real dEx_dy = (j < Ny - 1) ? (Ex[idx + Nz] - Ex[idx]) : (real)0.0;

    int in_x = (i < npml) || (i >= Nx - npml);
    int in_y = (j < npml) || (j >= Ny - npml);
    int in_z = (k < npml) || (k >= Nz - npml);

    real p_Hx_y = (real)0.0, p_Hx_z = (real)0.0;
    real p_Hy_x = (real)0.0, p_Hy_z = (real)0.0;
    real p_Hz_x = (real)0.0, p_Hz_y = (real)0.0;

    if (in_x) {
        int il = (i < npml) ? i : (npml + i - (Nx - npml));
        int xi = il * Ny * Nz + j * Nz + k;
        p_Hy_x = bx[i] * psi_Hy_x[xi] + cx[i] * dEz_dx;
        p_Hz_x = bx[i] * psi_Hz_x[xi] + cx[i] * dEy_dx;
        psi_Hy_x[xi] = p_Hy_x;
        psi_Hz_x[xi] = p_Hz_x;
    }
    if (in_y) {
        int jl = (j < npml) ? j : (npml + j - (Ny - npml));
        int yi = i * (2 * npml) * Nz + jl * Nz + k;
        p_Hx_y = by[j] * psi_Hx_y[yi] + cy[j] * dEz_dy;
        p_Hz_y = by[j] * psi_Hz_y[yi] + cy[j] * dEx_dy;
        psi_Hx_y[yi] = p_Hx_y;
        psi_Hz_y[yi] = p_Hz_y;
    }
    if (in_z) {
        int kl = (k < npml) ? k : (npml + k - (Nz - npml));
        int zi = i * Ny * (2 * npml) + j * (2 * npml) + kl;
        p_Hx_z = bz[k] * psi_Hx_z[zi] + cz[k] * dEy_dz;
        p_Hy_z = bz[k] * psi_Hy_z[zi] + cz[k] * dEx_dz;
        psi_Hx_z[zi] = p_Hx_z;
        psi_Hy_z[zi] = p_Hy_z;
    }

    Hx[idx] -= dtm * (dEz_dy * iky[j] + p_Hx_y - dEy_dz * ikz[k] - p_Hx_z);
    Hy[idx] -= dtm * (dEx_dz * ikz[k] + p_Hy_z - dEz_dx * ikx[i] - p_Hy_x);
    Hz[idx] -= dtm * (dEy_dx * ikx[i] + p_Hz_x - dEx_dy * iky[j] - p_Hz_y);
}

__kernel void update_E_interior(
    int Nx, int Ny, int Nz,
    int npml,
    real inv_dl,
    __global const real * restrict ce_x,  /* dt/(eps0*eps_r) at Ex edges */
    __global const real * restrict ce_y,
    __global const real * restrict ce_z,
    __global const real * restrict Hx,
    __global const real * restrict Hy,
    __global const real * restrict Hz,
    __global real * restrict Ex,
    __global real * restrict Ey,
    __global real * restrict Ez
) {
    int k = get_global_id(0) + npml;
    int j = get_global_id(1) + npml;
    int i = get_global_id(2) + npml;

    if (i >= Nx - npml || j >= Ny - npml || k >= Nz - npml) return;

    int idx = i * Ny * Nz + j * Nz + k;

    real dHz_dy = Hz[idx] - Hz[idx - Nz];
    real dHy_dz = Hy[idx] - Hy[idx - 1];
    real dHx_dz = Hx[idx] - Hx[idx - 1];
    real dHz_dx = Hz[idx] - Hz[idx - Ny * Nz];
    real dHy_dx = Hy[idx] - Hy[idx - Ny * Nz];
    real dHx_dy = Hx[idx] - Hx[idx - Nz];

    Ex[idx] += ce_x[idx] * inv_dl * (dHz_dy - dHy_dz);
    Ey[idx] += ce_y[idx] * inv_dl * (dHx_dz - dHz_dx);
    Ez[idx] += ce_z[idx] * inv_dl * (dHy_dx - dHx_dy);
}

__kernel void update_E_pml(
    int Nx, int Ny, int Nz,
    int npml,
    __global const real * restrict ce_x,  /* dt/(eps0*eps_r) at Ex edges */
    __global const real * restrict ce_y,
    __global const real * restrict ce_z,
    __global const real * restrict Hx,
    __global const real * restrict Hy,
    __global const real * restrict Hz,
    __global real * restrict Ex,
    __global real * restrict Ey,
    __global real * restrict Ez,
    /* ikx/iky/ikz hold 1/(kappa * dl), precomputed on host */
    __global const real * restrict bx, __global const real * restrict cx, __global const real * restrict ikx,
    __global const real * restrict by, __global const real * restrict cy, __global const real * restrict iky,
    __global const real * restrict bz, __global const real * restrict cz, __global const real * restrict ikz,
    __global real * restrict psi_Ex_y, __global real * restrict psi_Ex_z,
    __global real * restrict psi_Ey_x, __global real * restrict psi_Ey_z,
    __global real * restrict psi_Ez_x, __global real * restrict psi_Ez_y
) {
    int k = get_global_id(0);
    int j = get_global_id(1);
    int i = get_global_id(2);

    if (i >= Nx || j >= Ny || k >= Nz) return;

    if (npml > 0 &&
        i >= npml && i < Nx - npml &&
        j >= npml && j < Ny - npml &&
        k >= npml && k < Nz - npml) {
        return;
    }

    int idx = i * Ny * Nz + j * Nz + k;

    real dHz_dy = (j > 0) ? (Hz[idx] - Hz[idx - Nz]) : (real)0.0;
    real dHy_dz = (k > 0) ? (Hy[idx] - Hy[idx - 1])  : (real)0.0;
    real dHx_dz = (k > 0) ? (Hx[idx] - Hx[idx - 1])  : (real)0.0;
    real dHz_dx = (i > 0) ? (Hz[idx] - Hz[idx - Ny * Nz]) : (real)0.0;
    real dHy_dx = (i > 0) ? (Hy[idx] - Hy[idx - Ny * Nz]) : (real)0.0;
    real dHx_dy = (j > 0) ? (Hx[idx] - Hx[idx - Nz]) : (real)0.0;

    int in_x = (i < npml) || (i >= Nx - npml);
    int in_y = (j < npml) || (j >= Ny - npml);
    int in_z = (k < npml) || (k >= Nz - npml);

    real p_Ex_y = (real)0.0, p_Ex_z = (real)0.0;
    real p_Ey_x = (real)0.0, p_Ey_z = (real)0.0;
    real p_Ez_x = (real)0.0, p_Ez_y = (real)0.0;

    if (in_x) {
        int il = (i < npml) ? i : (npml + i - (Nx - npml));
        int xi = il * Ny * Nz + j * Nz + k;
        p_Ey_x = bx[i] * psi_Ey_x[xi] + cx[i] * dHz_dx;
        p_Ez_x = bx[i] * psi_Ez_x[xi] + cx[i] * dHy_dx;
        psi_Ey_x[xi] = p_Ey_x;
        psi_Ez_x[xi] = p_Ez_x;
    }
    if (in_y) {
        int jl = (j < npml) ? j : (npml + j - (Ny - npml));
        int yi = i * (2 * npml) * Nz + jl * Nz + k;
        p_Ex_y = by[j] * psi_Ex_y[yi] + cy[j] * dHz_dy;
        p_Ez_y = by[j] * psi_Ez_y[yi] + cy[j] * dHx_dy;
        psi_Ex_y[yi] = p_Ex_y;
        psi_Ez_y[yi] = p_Ez_y;
    }
    if (in_z) {
        int kl = (k < npml) ? k : (npml + k - (Nz - npml));
        int zi = i * Ny * (2 * npml) + j * (2 * npml) + kl;
        p_Ex_z = bz[k] * psi_Ex_z[zi] + cz[k] * dHy_dz;
        p_Ey_z = bz[k] * psi_Ey_z[zi] + cz[k] * dHx_dz;
        psi_Ex_z[zi] = p_Ex_z;
        psi_Ey_z[zi] = p_Ey_z;
    }

    Ex[idx] += ce_x[idx] * (dHz_dy * iky[j] + p_Ex_y - dHy_dz * ikz[k] - p_Ex_z);
    Ey[idx] += ce_y[idx] * (dHx_dz * ikz[k] + p_Ey_z - dHz_dx * ikx[i] - p_Ey_x);
    Ez[idx] += ce_z[idx] * (dHy_dx * ikx[i] + p_Ez_x - dHx_dy * iky[j] - p_Ez_y);
}
