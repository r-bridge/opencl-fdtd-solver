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
    float dl, float dtm,
    __global const float *Ex,
    __global const float *Ey,
    __global const float *Ez,
    __global float *Hx,
    __global float *Hy,
    __global float *Hz
) {
    int k = get_global_id(0) + npml;
    int j = get_global_id(1) + npml;
    int i = get_global_id(2) + npml;

    if (i >= Nx - npml || j >= Ny - npml || k >= Nz - npml) return;

    int idx = i * Ny * Nz + j * Nz + k;
    float inv_dl = 1.0f / dl;

    float dEz_dy = Ez[idx + Nz] - Ez[idx];
    float dEy_dz = Ey[idx + 1] - Ey[idx];
    float dEx_dz = Ex[idx + 1] - Ex[idx];
    float dEz_dx = Ez[idx + Ny * Nz] - Ez[idx];
    float dEy_dx = Ey[idx + Ny * Nz] - Ey[idx];
    float dEx_dy = Ex[idx + Nz] - Ex[idx];

    Hx[idx] -= dtm * (dEz_dy - dEy_dz) * inv_dl;
    Hy[idx] -= dtm * (dEx_dz - dEz_dx) * inv_dl;
    Hz[idx] -= dtm * (dEy_dx - dEx_dy) * inv_dl;
}

__kernel void update_H_pml(
    int Nx, int Ny, int Nz,
    int npml,
    float dl, float dtm,
    __global const float *Ex,
    __global const float *Ey,
    __global const float *Ez,
    __global float *Hx,
    __global float *Hy,
    __global float *Hz,
    __global const float *bx, __global const float *cx, __global const float *kx,
    __global const float *by, __global const float *cy, __global const float *ky,
    __global const float *bz, __global const float *cz, __global const float *kz,
    __global float *psi_Hx_y, __global float *psi_Hx_z,
    __global float *psi_Hy_x, __global float *psi_Hy_z,
    __global float *psi_Hz_x, __global float *psi_Hz_y
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

    float dEz_dy = (j < Ny - 1) ? (Ez[idx + Nz] - Ez[idx]) : 0.0f;
    float dEy_dz = (k < Nz - 1) ? (Ey[idx + 1] - Ey[idx])  : 0.0f;
    float dEx_dz = (k < Nz - 1) ? (Ex[idx + 1] - Ex[idx])  : 0.0f;
    float dEz_dx = (i < Nx - 1) ? (Ez[idx + Ny * Nz] - Ez[idx]) : 0.0f;
    float dEy_dx = (i < Nx - 1) ? (Ey[idx + Ny * Nz] - Ey[idx]) : 0.0f;
    float dEx_dy = (j < Ny - 1) ? (Ex[idx + Nz] - Ex[idx]) : 0.0f;

    int in_x = (i < npml) || (i >= Nx - npml);
    int in_y = (j < npml) || (j >= Ny - npml);
    int in_z = (k < npml) || (k >= Nz - npml);

    float p_Hx_y = 0.0f, p_Hx_z = 0.0f;
    float p_Hy_x = 0.0f, p_Hy_z = 0.0f;
    float p_Hz_x = 0.0f, p_Hz_y = 0.0f;

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

    Hx[idx] -= dtm * (dEz_dy / (ky[j] * dl) + p_Hx_y - dEy_dz / (kz[k] * dl) - p_Hx_z);
    Hy[idx] -= dtm * (dEx_dz / (kz[k] * dl) + p_Hy_z - dEz_dx / (kx[i] * dl) - p_Hy_x);
    Hz[idx] -= dtm * (dEy_dx / (kx[i] * dl) + p_Hz_x - dEx_dy / (ky[j] * dl) - p_Hz_y);
}

__kernel void update_E_interior(
    int Nx, int Ny, int Nz,
    int npml,
    float dl, float dt,
    float eps0,
    __global const float *eps_r,
    __global const float *Hx,
    __global const float *Hy,
    __global const float *Hz,
    __global float *Ex,
    __global float *Ey,
    __global float *Ez
) {
    int k = get_global_id(0) + npml;
    int j = get_global_id(1) + npml;
    int i = get_global_id(2) + npml;

    if (i >= Nx - npml || j >= Ny - npml || k >= Nz - npml) return;

    int idx = i * Ny * Nz + j * Nz + k;
    float inv_dl = 1.0f / dl;

    float dHz_dy = Hz[idx] - Hz[idx - Nz];
    float dHy_dz = Hy[idx] - Hy[idx - 1];
    float dHx_dz = Hx[idx] - Hx[idx - 1];
    float dHz_dx = Hz[idx] - Hz[idx - Ny * Nz];
    float dHy_dx = Hy[idx] - Hy[idx - Ny * Nz];
    float dHx_dy = Hx[idx] - Hx[idx - Nz];

    float coeff = dt / (eps0 * eps_r[idx]) * inv_dl;
    Ex[idx] += coeff * (dHz_dy - dHy_dz);
    Ey[idx] += coeff * (dHx_dz - dHz_dx);
    Ez[idx] += coeff * (dHy_dx - dHx_dy);
}

__kernel void update_E_pml(
    int Nx, int Ny, int Nz,
    int npml,
    float dl, float dt,
    float eps0,
    __global const float *eps_r,
    __global const float *Hx,
    __global const float *Hy,
    __global const float *Hz,
    __global float *Ex,
    __global float *Ey,
    __global float *Ez,
    __global const float *bx, __global const float *cx, __global const float *kx,
    __global const float *by, __global const float *cy, __global const float *ky,
    __global const float *bz, __global const float *cz, __global const float *kz,
    __global float *psi_Ex_y, __global float *psi_Ex_z,
    __global float *psi_Ey_x, __global float *psi_Ey_z,
    __global float *psi_Ez_x, __global float *psi_Ez_y
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

    float dHz_dy = (j > 0) ? (Hz[idx] - Hz[idx - Nz]) : 0.0f;
    float dHy_dz = (k > 0) ? (Hy[idx] - Hy[idx - 1])  : 0.0f;
    float dHx_dz = (k > 0) ? (Hx[idx] - Hx[idx - 1])  : 0.0f;
    float dHz_dx = (i > 0) ? (Hz[idx] - Hz[idx - Ny * Nz]) : 0.0f;
    float dHy_dx = (i > 0) ? (Hy[idx] - Hy[idx - Ny * Nz]) : 0.0f;
    float dHx_dy = (j > 0) ? (Hx[idx] - Hx[idx - Nz]) : 0.0f;

    int in_x = (i < npml) || (i >= Nx - npml);
    int in_y = (j < npml) || (j >= Ny - npml);
    int in_z = (k < npml) || (k >= Nz - npml);

    float p_Ex_y = 0.0f, p_Ex_z = 0.0f;
    float p_Ey_x = 0.0f, p_Ey_z = 0.0f;
    float p_Ez_x = 0.0f, p_Ez_y = 0.0f;

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

    float coeff = dt / (eps0 * eps_r[idx]);
    Ex[idx] += coeff * (dHz_dy / (ky[j] * dl) + p_Ex_y - dHy_dz / (kz[k] * dl) - p_Ex_z);
    Ey[idx] += coeff * (dHx_dz / (kz[k] * dl) + p_Ey_z - dHz_dx / (kx[i] * dl) - p_Ey_x);
    Ez[idx] += coeff * (dHy_dx / (kx[i] * dl) + p_Ez_x - dHx_dy / (ky[j] * dl) - p_Ez_y);
}
