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
    float dtm_dl,  /* dt / (mu0 * dl), folded on host */
    __global const float * restrict Ex,
    __global const float * restrict Ey,
    __global const float * restrict Ez,
    __global float * restrict Hx,
    __global float * restrict Hy,
    __global float * restrict Hz
) {
    int k = get_global_id(0) + npml;
    int j = get_global_id(1) + npml;
    int i = get_global_id(2) + npml;

    if (i >= Nx - npml || j >= Ny - npml || k >= Nz - npml) return;

    int idx = i * Ny * Nz + j * Nz + k;

    float dEz_dy = Ez[idx + Nz] - Ez[idx];
    float dEy_dz = Ey[idx + 1] - Ey[idx];
    float dEx_dz = Ex[idx + 1] - Ex[idx];
    float dEz_dx = Ez[idx + Ny * Nz] - Ez[idx];
    float dEy_dx = Ey[idx + Ny * Nz] - Ey[idx];
    float dEx_dy = Ex[idx + Nz] - Ex[idx];

    Hx[idx] -= dtm_dl * (dEz_dy - dEy_dz);
    Hy[idx] -= dtm_dl * (dEx_dz - dEz_dx);
    Hz[idx] -= dtm_dl * (dEy_dx - dEx_dy);
}

/* Map a packed shell index onto (i,j,k) via slab metadata:
 * slabs[7*s + 0..6] = start offset, i0, j0, k0, di, dj, dk.
 * Slabs partition the PML shell (or the full grid when npml == 0), so one
 * 1-D launch covers exactly the shell cells with no interior no-op threads. */
inline int shell_ijk(
    int t, int n_slabs,
    __global const int * restrict slabs,
    int *i, int *j, int *k
) {
    int s = 0;
    for (int q = 1; q < n_slabs; ++q) {
        if (slabs[7 * q] <= t) s = q;
    }
    int base = 7 * s;
    int loc = t - slabs[base + 0];
    int dj = slabs[base + 5];
    int dk = slabs[base + 6];
    int jq = loc / dk;
    *k = slabs[base + 3] + (loc - jq * dk);
    int iq = jq / dj;
    *j = slabs[base + 2] + (jq - iq * dj);
    *i = slabs[base + 1] + iq;
    return s;
}

__kernel void update_H_pml(
    int Nx, int Ny, int Nz,
    int npml,
    int n_shell, int n_slabs,
    __global const int * restrict slabs,
    float dtm,
    __global const float * restrict Ex,
    __global const float * restrict Ey,
    __global const float * restrict Ez,
    __global float * restrict Hx,
    __global float * restrict Hy,
    __global float * restrict Hz,
    /* ikx/iky/ikz hold 1/(kappa * dl), precomputed on host */
    __global const float * restrict bx, __global const float * restrict cx, __global const float * restrict ikx,
    __global const float * restrict by, __global const float * restrict cy, __global const float * restrict iky,
    __global const float * restrict bz, __global const float * restrict cz, __global const float * restrict ikz,
    __global float * restrict psi_Hx_y, __global float * restrict psi_Hx_z,
    __global float * restrict psi_Hy_x, __global float * restrict psi_Hy_z,
    __global float * restrict psi_Hz_x, __global float * restrict psi_Hz_y
) {
    int t = get_global_id(0);
    if (t >= n_shell) return;
    int i, j, k;
    shell_ijk(t, n_slabs, slabs, &i, &j, &k);

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

    Hx[idx] -= dtm * (dEz_dy * iky[j] + p_Hx_y - dEy_dz * ikz[k] - p_Hx_z);
    Hy[idx] -= dtm * (dEx_dz * ikz[k] + p_Hy_z - dEz_dx * ikx[i] - p_Hy_x);
    Hz[idx] -= dtm * (dEy_dx * ikx[i] + p_Hz_x - dEx_dy * iky[j] - p_Hz_y);
}

__kernel void update_E_interior(
    int Nx, int Ny, int Nz,
    int npml,
    float inv_dl,
    __global const float * restrict ce,  /* dt / (eps0 * eps_r), precomputed */
    __global const float * restrict Hx,
    __global const float * restrict Hy,
    __global const float * restrict Hz,
    __global float * restrict Ex,
    __global float * restrict Ey,
    __global float * restrict Ez
) {
    int k = get_global_id(0) + npml;
    int j = get_global_id(1) + npml;
    int i = get_global_id(2) + npml;

    if (i >= Nx - npml || j >= Ny - npml || k >= Nz - npml) return;

    int idx = i * Ny * Nz + j * Nz + k;

    float dHz_dy = Hz[idx] - Hz[idx - Nz];
    float dHy_dz = Hy[idx] - Hy[idx - 1];
    float dHx_dz = Hx[idx] - Hx[idx - 1];
    float dHz_dx = Hz[idx] - Hz[idx - Ny * Nz];
    float dHy_dx = Hy[idx] - Hy[idx - Ny * Nz];
    float dHx_dy = Hx[idx] - Hx[idx - Nz];

    float coeff = ce[idx] * inv_dl;
    Ex[idx] += coeff * (dHz_dy - dHy_dz);
    Ey[idx] += coeff * (dHx_dz - dHz_dx);
    Ez[idx] += coeff * (dHy_dx - dHx_dy);
}

/* Same packed-shell mapping as update_H_pml. */
__kernel void update_E_pml(
    int Nx, int Ny, int Nz,
    int npml,
    int n_shell, int n_slabs,
    __global const int * restrict slabs,
    __global const float * restrict ce,  /* dt / (eps0 * eps_r), precomputed */
    __global const float * restrict Hx,
    __global const float * restrict Hy,
    __global const float * restrict Hz,
    __global float * restrict Ex,
    __global float * restrict Ey,
    __global float * restrict Ez,
    /* ikx/iky/ikz hold 1/(kappa * dl), precomputed on host */
    __global const float * restrict bx, __global const float * restrict cx, __global const float * restrict ikx,
    __global const float * restrict by, __global const float * restrict cy, __global const float * restrict iky,
    __global const float * restrict bz, __global const float * restrict cz, __global const float * restrict ikz,
    __global float * restrict psi_Ex_y, __global float * restrict psi_Ex_z,
    __global float * restrict psi_Ey_x, __global float * restrict psi_Ey_z,
    __global float * restrict psi_Ez_x, __global float * restrict psi_Ez_y
) {
    int t = get_global_id(0);
    if (t >= n_shell) return;
    int i, j, k;
    shell_ijk(t, n_slabs, slabs, &i, &j, &k);

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

    float coeff = ce[idx];
    Ex[idx] += coeff * (dHz_dy * iky[j] + p_Ex_y - dHy_dz * ikz[k] - p_Ex_z);
    Ey[idx] += coeff * (dHx_dz * ikz[k] + p_Ey_z - dHz_dx * ikx[i] - p_Ey_x);
    Ez[idx] += coeff * (dHy_dx * ikx[i] + p_Ez_x - dHx_dy * iky[j] - p_Ez_y);
}
