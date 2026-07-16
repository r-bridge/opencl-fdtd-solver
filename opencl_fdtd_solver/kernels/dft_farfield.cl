/* Copyright (C) 2026: OpenCL FDTD Solver Contributors
 * Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
 *
 * This file is part of opencl-fdtd-solver (GPLv3 or later).
 */

/* Legacy full-box DFT (volume buffer); prefer accumulate_dft_face. */
__kernel void accumulate_dft(
    int Nx, int Ny, int Nz,
    int ix0, int ix1,
    int iy0, int iy1,
    int iz0, int iz1,
    float phase_real, float phase_imag,
    __global const float *field,
    __global float2 *field_dft
) {
    int k = get_global_id(0);
    int j = get_global_id(1);
    int i = get_global_id(2);

    int x_dim = ix1 - ix0 + 1;
    int y_dim = iy1 - iy0 + 1;
    int z_dim = iz1 - iz0 + 1;

    if (i >= x_dim || j >= y_dim || k >= z_dim) return;

    int abs_i = ix0 + i;
    int abs_j = iy0 + j;
    int abs_k = iz0 + k;

    if (abs_i == ix0 || abs_i == ix1 || abs_j == iy0 || abs_j == iy1 || abs_k == iz0 || abs_k == iz1) {
        int idx = abs_i * Ny * Nz + abs_j * Nz + abs_k;
        float val = field[idx];

        float2 current_dft = field_dft[idx];
        current_dft.x += val * phase_real;
        current_dft.y += val * phase_imag;
        field_dft[idx] = current_dft;
    }
}

/*
 * Accumulate DFT onto one Huygens face into a packed float2 buffer.
 * face_id: 0=x0, 1=x1, 2=y0, 3=y1, 4=z0, 5=z1
 * Work size: (u, v) with u along the first face axis, v the second
 *   x-faces: (nzf, nyf) → abs (ix, iy0+v, iz0+u)
 *   y-faces: (nzf, nxf) → abs (ix0+v, iy, iz0+u)
 *   z-faces: (nyf, nxf) → abs (ix0+v, iy0+u, iz)
 */
__kernel void accumulate_dft_face(
    int Nx, int Ny, int Nz,
    int face_id,
    int ix0, int ix1,
    int iy0, int iy1,
    int iz0, int iz1,
    int face_offset,
    float phase_real, float phase_imag,
    __global const float *field,
    __global float2 *face_dft
) {
    int u = get_global_id(0);
    int v = get_global_id(1);
    int abs_i, abs_j, abs_k;
    int nxf = ix1 - ix0 + 1;
    int nyf = iy1 - iy0 + 1;
    int nzf = iz1 - iz0 + 1;
    int face_li;

    if (face_id == 0 || face_id == 1) {
        if (u >= nzf || v >= nyf) return;
        abs_i = (face_id == 0) ? ix0 : ix1;
        abs_j = iy0 + v;
        abs_k = iz0 + u;
        face_li = v * nzf + u;
    } else if (face_id == 2 || face_id == 3) {
        if (u >= nzf || v >= nxf) return;
        abs_i = ix0 + v;
        abs_j = (face_id == 2) ? iy0 : iy1;
        abs_k = iz0 + u;
        face_li = v * nzf + u;
    } else {
        if (u >= nyf || v >= nxf) return;
        abs_i = ix0 + v;
        abs_j = iy0 + u;
        abs_k = (face_id == 4) ? iz0 : iz1;
        face_li = v * nyf + u;
    }

    int idx = abs_i * Ny * Nz + abs_j * Nz + abs_k;
    float val = field[idx];
    int dst = face_offset + face_li;
    float2 cur = face_dft[dst];
    cur.x += val * phase_real;
    cur.y += val * phase_imag;
    face_dft[dst] = cur;
}

inline void dft_add(__global float2 *slot, float val, float pr, float pi) {
    float2 cur = *slot;
    cur.x += val * pr;
    cur.y += val * pi;
    *slot = cur;
}

/*
 * One launch over packed face samples: for each Huygens face sample,
 * DFT only the tangential Ex..Hz components that enter the N/L integral.
 * Replaces 36 per-step accumulate_dft_face launches.
 */
__kernel void accumulate_dft_faces_fused(
    int Nx, int Ny, int Nz,
    int ix0, int ix1,
    int iy0, int iy1,
    int iz0, int iz1,
    int off0, int off1, int off2, int off3, int off4, int off5,
    int n_face,
    float phase_real, float phase_imag,
    __global const float * restrict Ex,
    __global const float * restrict Ey,
    __global const float * restrict Ez,
    __global const float * restrict Hx,
    __global const float * restrict Hy,
    __global const float * restrict Hz,
    __global float2 * restrict Ex_dft,
    __global float2 * restrict Ey_dft,
    __global float2 * restrict Ez_dft,
    __global float2 * restrict Hx_dft,
    __global float2 * restrict Hy_dft,
    __global float2 * restrict Hz_dft
) {
    int face_i = get_global_id(0);
    if (face_i >= n_face) return;

    int nxf = ix1 - ix0 + 1;
    int nyf = iy1 - iy0 + 1;
    int nzf = iz1 - iz0 + 1;
    int face, loc, abs_i, abs_j, abs_k;

    if (face_i < off1) {
        face = 0; loc = face_i - off0;
        abs_i = ix0; abs_j = iy0 + loc / nzf; abs_k = iz0 + (loc - (loc / nzf) * nzf);
    } else if (face_i < off2) {
        face = 1; loc = face_i - off1;
        abs_i = ix1; abs_j = iy0 + loc / nzf; abs_k = iz0 + (loc - (loc / nzf) * nzf);
    } else if (face_i < off3) {
        face = 2; loc = face_i - off2;
        abs_i = ix0 + loc / nzf; abs_j = iy0; abs_k = iz0 + (loc - (loc / nzf) * nzf);
    } else if (face_i < off4) {
        face = 3; loc = face_i - off3;
        abs_i = ix0 + loc / nzf; abs_j = iy1; abs_k = iz0 + (loc - (loc / nzf) * nzf);
    } else if (face_i < off5) {
        face = 4; loc = face_i - off4;
        abs_i = ix0 + loc / nyf; abs_j = iy0 + (loc - (loc / nyf) * nyf); abs_k = iz0;
    } else {
        face = 5; loc = face_i - off5;
        abs_i = ix0 + loc / nyf; abs_j = iy0 + (loc - (loc / nyf) * nyf); abs_k = iz1;
    }
    (void)nxf;

    int idx = abs_i * Ny * Nz + abs_j * Nz + abs_k;
    float pr = phase_real, pi = phase_imag;

/* Tangential only (same components as face_sample_NL). */
    if (face <= 1) {
        dft_add(&Ey_dft[face_i], Ey[idx], pr, pi);
        dft_add(&Ez_dft[face_i], Ez[idx], pr, pi);
        dft_add(&Hy_dft[face_i], Hy[idx], pr, pi);
        dft_add(&Hz_dft[face_i], Hz[idx], pr, pi);
    } else if (face <= 3) {
        dft_add(&Ex_dft[face_i], Ex[idx], pr, pi);
        dft_add(&Ez_dft[face_i], Ez[idx], pr, pi);
        dft_add(&Hx_dft[face_i], Hx[idx], pr, pi);
        dft_add(&Hz_dft[face_i], Hz[idx], pr, pi);
    } else {
        dft_add(&Ex_dft[face_i], Ex[idx], pr, pi);
        dft_add(&Ey_dft[face_i], Ey[idx], pr, pi);
        dft_add(&Hx_dft[face_i], Hx[idx], pr, pi);
        dft_add(&Hy_dft[face_i], Hy[idx], pr, pi);
    }
}

/*
 * Relative L2 change of face DFT vs a previous snapshot:
 *   sqrt(sum |cur-prev|^2 / sum |cur|^2)
 * Workgroup-local reduction; host sums partials (tiny download).
 */
__kernel void dft_rel_change_partial(
    int n,
    __global const float2 * restrict Ex_c,
    __global const float2 * restrict Ey_c,
    __global const float2 * restrict Ez_c,
    __global const float2 * restrict Hx_c,
    __global const float2 * restrict Hy_c,
    __global const float2 * restrict Hz_c,
    __global const float2 * restrict Ex_p,
    __global const float2 * restrict Ey_p,
    __global const float2 * restrict Ez_p,
    __global const float2 * restrict Hx_p,
    __global const float2 * restrict Hy_p,
    __global const float2 * restrict Hz_p,
    __global float * restrict partial_num,
    __global float * restrict partial_den,
    __local float *sn,
    __local float *sd
) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int grp = get_group_id(0);
    float nsum = 0.0f;
    float dsum = 0.0f;
    if (gid < n) {
        float2 c, p, d;
        #define DFT_ACC(CUR, PREV) \
            c = (CUR)[gid]; p = (PREV)[gid]; \
            d.x = c.x - p.x; d.y = c.y - p.y; \
            nsum += d.x * d.x + d.y * d.y; \
            dsum += c.x * c.x + c.y * c.y;
        DFT_ACC(Ex_c, Ex_p);
        DFT_ACC(Ey_c, Ey_p);
        DFT_ACC(Ez_c, Ez_p);
        DFT_ACC(Hx_c, Hx_p);
        DFT_ACC(Hy_c, Hy_p);
        DFT_ACC(Hz_c, Hz_p);
        #undef DFT_ACC
    }
    sn[lid] = nsum;
    sd[lid] = dsum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            sn[lid] += sn[lid + stride];
            sd[lid] += sd[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        partial_num[grp] = sn[0];
        partial_den[grp] = sd[0];
    }
}

inline float2 cmul(float2 a, float2 b) {
    return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline void caccum(float2 *acc, float2 val) {
    acc->x += val.x;
    acc->y += val.y;
}

inline void atomic_add_float(__global volatile float *addr, float val) {
    union { unsigned int u; float f; } oldv, newv;
    do {
        oldv.f = *addr;
        newv.f = oldv.f + val;
    } while (atomic_cmpxchg(
                 (__global volatile unsigned int *)addr, oldv.u, newv.u)
             != oldv.u);
}

inline void atomic_add_float2(__global float2 *addr, float2 val) {
    __global float *p = (__global float *)addr;
    atomic_add_float(&p[0], val.x);
    atomic_add_float(&p[1], val.y);
}

/* Contribution of one packed face sample into N,L (6 float2). */
inline void face_sample_NL(
    int face_i,
    float rx, float ry, float rz,
    float k_wave, float dA, float dl,
    int ix0, int ix1, int iy0, int iy1, int iz0, int iz1,
    int off0, int off1, int off2, int off3, int off4, int off5,
    __global const float2 *Ex_f,
    __global const float2 *Ey_f,
    __global const float2 *Ez_f,
    __global const float2 *Hx_f,
    __global const float2 *Hy_f,
    __global const float2 *Hz_f,
    float2 *Nx, float2 *Ny, float2 *Nz,
    float2 *Lx, float2 *Ly, float2 *Lz
) {
    int nxf = ix1 - ix0 + 1;
    int nyf = iy1 - iy0 + 1;
    int nzf = iz1 - iz0 + 1;
    int face, loc, abs_i, abs_j, abs_k;
    float nf, xp, yp, zp;

    if (face_i < off1) {
        face = 0; loc = face_i - off0; nf = -1.0f;
        abs_i = ix0; abs_j = iy0 + loc / nzf; abs_k = iz0 + (loc - (loc / nzf) * nzf);
    } else if (face_i < off2) {
        face = 1; loc = face_i - off1; nf = 1.0f;
        abs_i = ix1; abs_j = iy0 + loc / nzf; abs_k = iz0 + (loc - (loc / nzf) * nzf);
    } else if (face_i < off3) {
        face = 2; loc = face_i - off2; nf = -1.0f;
        abs_i = ix0 + loc / nzf; abs_j = iy0; abs_k = iz0 + (loc - (loc / nzf) * nzf);
    } else if (face_i < off4) {
        face = 3; loc = face_i - off3; nf = 1.0f;
        abs_i = ix0 + loc / nzf; abs_j = iy1; abs_k = iz0 + (loc - (loc / nzf) * nzf);
    } else if (face_i < off5) {
        face = 4; loc = face_i - off4; nf = -1.0f;
        abs_i = ix0 + loc / nyf; abs_j = iy0 + (loc - (loc / nyf) * nyf); abs_k = iz0;
    } else {
        face = 5; loc = face_i - off5; nf = 1.0f;
        abs_i = ix0 + loc / nyf; abs_j = iy0 + (loc - (loc / nyf) * nyf); abs_k = iz1;
    }
    (void)nxf;
    xp = abs_i * dl; yp = abs_j * dl; zp = abs_k * dl;
    int li = face_i;
    float phase = k_wave * (rx * xp + ry * yp + rz * zp);
    float2 ph = (float2)(cos(phase), sin(phase));

    if (face <= 1) {
        float2 Jy = cmul((float2)( nf * Hz_f[li].x,  nf * Hz_f[li].y), ph);
        float2 Jz = cmul((float2)(-nf * Hy_f[li].x, -nf * Hy_f[li].y), ph);
        float2 My = cmul((float2)(-nf * Ez_f[li].x, -nf * Ez_f[li].y), ph);
        float2 Mz = cmul((float2)( nf * Ey_f[li].x,  nf * Ey_f[li].y), ph);
        caccum(Ny, (float2)(Jy.x * dA, Jy.y * dA));
        caccum(Nz, (float2)(Jz.x * dA, Jz.y * dA));
        caccum(Ly, (float2)(My.x * dA, My.y * dA));
        caccum(Lz, (float2)(Mz.x * dA, Mz.y * dA));
    } else if (face <= 3) {
        float2 Jx = cmul((float2)(-nf * Hz_f[li].x, -nf * Hz_f[li].y), ph);
        float2 Jz = cmul((float2)( nf * Hx_f[li].x,  nf * Hx_f[li].y), ph);
        float2 Mx = cmul((float2)( nf * Ez_f[li].x,  nf * Ez_f[li].y), ph);
        float2 Mz = cmul((float2)(-nf * Ex_f[li].x, -nf * Ex_f[li].y), ph);
        caccum(Nx, (float2)(Jx.x * dA, Jx.y * dA));
        caccum(Nz, (float2)(Jz.x * dA, Jz.y * dA));
        caccum(Lx, (float2)(Mx.x * dA, Mx.y * dA));
        caccum(Lz, (float2)(Mz.x * dA, Mz.y * dA));
    } else {
        float2 Jx = cmul((float2)( nf * Hy_f[li].x,  nf * Hy_f[li].y), ph);
        float2 Jy = cmul((float2)(-nf * Hx_f[li].x, -nf * Hx_f[li].y), ph);
        float2 Mx = cmul((float2)(-nf * Ey_f[li].x, -nf * Ey_f[li].y), ph);
        float2 My = cmul((float2)( nf * Ex_f[li].x,  nf * Ex_f[li].y), ph);
        caccum(Nx, (float2)(Jx.x * dA, Jx.y * dA));
        caccum(Ny, (float2)(Jy.x * dA, Jy.y * dA));
        caccum(Lx, (float2)(Mx.x * dA, Mx.y * dA));
        caccum(Ly, (float2)(My.x * dA, My.y * dA));
    }
}

/*
 * Parallel over face samples (dim0) and observation points (dim1).
 * Local reduction along dim0, then atomic add into NL[obs*6 + c].
 */
__kernel void farfield_accumulate_nl(
    int n_face,
    int n_obs,
    __global const float *obs_xyz,
    float k_wave,
    float dl,
    int ix0, int ix1,
    int iy0, int iy1,
    int iz0, int iz1,
    int off0, int off1, int off2, int off3, int off4, int off5,
    __global const float2 *Ex_f,
    __global const float2 *Ey_f,
    __global const float2 *Ez_f,
    __global const float2 *Hx_f,
    __global const float2 *Hy_f,
    __global const float2 *Hz_f,
    __global float2 *NL_out,
    __local float2 *scratch
) {
    int face_i = get_global_id(0);
    int obs = get_global_id(1);
    int lid = get_local_id(0);
    int lsize = get_local_size(0);

    float2 Nx = (float2)(0.0f, 0.0f);
    float2 Ny = (float2)(0.0f, 0.0f);
    float2 Nz = (float2)(0.0f, 0.0f);
    float2 Lx = (float2)(0.0f, 0.0f);
    float2 Ly = (float2)(0.0f, 0.0f);
    float2 Lz = (float2)(0.0f, 0.0f);

    if (face_i < n_face && obs < n_obs) {
        float ox = obs_xyz[3 * obs + 0];
        float oy = obs_xyz[3 * obs + 1];
        float oz = obs_xyz[3 * obs + 2];
        float r = sqrt(ox * ox + oy * oy + oz * oz);
        if (r < 1.0e-30f) r = 1.0e-30f;
        float rx = ox / r, ry = oy / r, rz = oz / r;
        float dA = dl * dl;
        face_sample_NL(
            face_i, rx, ry, rz, k_wave, dA, dl,
            ix0, ix1, iy0, iy1, iz0, iz1,
            off0, off1, off2, off3, off4, off5,
            Ex_f, Ey_f, Ez_f, Hx_f, Hy_f, Hz_f,
            &Nx, &Ny, &Nz, &Lx, &Ly, &Lz);
    }

/* scratch layout: [comp][lid], comp=0..5 */
    scratch[0 * lsize + lid] = Nx;
    scratch[1 * lsize + lid] = Ny;
    scratch[2 * lsize + lid] = Nz;
    scratch[3 * lsize + lid] = Lx;
    scratch[4 * lsize + lid] = Ly;
    scratch[5 * lsize + lid] = Lz;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            for (int c = 0; c < 6; c++) {
                int a = c * lsize + lid;
                scratch[a].x += scratch[a + stride].x;
                scratch[a].y += scratch[a + stride].y;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0 && obs < n_obs) {
        int base = 6 * obs;
        atomic_add_float2(&NL_out[base + 0], scratch[0 * lsize]);
        atomic_add_float2(&NL_out[base + 1], scratch[1 * lsize]);
        atomic_add_float2(&NL_out[base + 2], scratch[2 * lsize]);
        atomic_add_float2(&NL_out[base + 3], scratch[3 * lsize]);
        atomic_add_float2(&NL_out[base + 4], scratch[4 * lsize]);
        atomic_add_float2(&NL_out[base + 5], scratch[5 * lsize]);
    }
}

/* Convert integrated N,L → far E,H for each observation. */
__kernel void farfield_nl_to_eh(
    int n_obs,
    __global const float *obs_xyz,
    float k_wave,
    float eta0,
    __global const float2 *NL_in,
    __global float2 *EH_out
) {
    int p = get_global_id(0);
    if (p >= n_obs) return;

    float ox = obs_xyz[3 * p + 0];
    float oy = obs_xyz[3 * p + 1];
    float oz = obs_xyz[3 * p + 2];
    float r = sqrt(ox * ox + oy * oy + oz * oz);
    if (r < 1.0e-30f) r = 1.0e-30f;
    float rx = ox / r, ry = oy / r, rz = oz / r;
    float rhat[3] = { rx, ry, rz };

    float2 Nvec[3] = { NL_in[6 * p + 0], NL_in[6 * p + 1], NL_in[6 * p + 2] };
    float2 Lvec[3] = { NL_in[6 * p + 3], NL_in[6 * p + 4], NL_in[6 * p + 5] };

    float ang = k_wave * r;
    // Outgoing wave ~ e^{-j k r}
    float2 eikr = (float2)(cos(ang), -sin(ang));
    float scale = k_wave / (4.0f * 3.14159265358979323846f * r);
    float2 pref = cmul((float2)(0.0f, -scale), eikr);

    float2 rxL[3], Nt[3], E[3], H[3];
    rxL[0] = (float2)(rhat[1] * Lvec[2].x - rhat[2] * Lvec[1].x,
                      rhat[1] * Lvec[2].y - rhat[2] * Lvec[1].y);
    rxL[1] = (float2)(rhat[2] * Lvec[0].x - rhat[0] * Lvec[2].x,
                      rhat[2] * Lvec[0].y - rhat[0] * Lvec[2].y);
    rxL[2] = (float2)(rhat[0] * Lvec[1].x - rhat[1] * Lvec[0].x,
                      rhat[0] * Lvec[1].y - rhat[1] * Lvec[0].y);

    float2 Ndot = (float2)(
        rhat[0] * Nvec[0].x + rhat[1] * Nvec[1].x + rhat[2] * Nvec[2].x,
        rhat[0] * Nvec[0].y + rhat[1] * Nvec[1].y + rhat[2] * Nvec[2].y);

    for (int c = 0; c < 3; c++) {
        Nt[c] = (float2)(Nvec[c].x - Ndot.x * rhat[c], Nvec[c].y - Ndot.y * rhat[c]);
        float2 tE = (float2)(eta0 * Nt[c].x + rxL[c].x, eta0 * Nt[c].y + rxL[c].y);
        E[c] = cmul(pref, tE);
    }
    // Far-field TEM: H = -r̂ × E / η
    H[0] = (float2)(-(rhat[1] * E[2].x - rhat[2] * E[1].x) / eta0,
                    -(rhat[1] * E[2].y - rhat[2] * E[1].y) / eta0);
    H[1] = (float2)(-(rhat[2] * E[0].x - rhat[0] * E[2].x) / eta0,
                    -(rhat[2] * E[0].y - rhat[0] * E[2].y) / eta0);
    H[2] = (float2)(-(rhat[0] * E[1].x - rhat[1] * E[0].x) / eta0,
                    -(rhat[0] * E[1].y - rhat[1] * E[0].y) / eta0);

    int o = 6 * p;
    EH_out[o + 0] = E[0];
    EH_out[o + 1] = E[1];
    EH_out[o + 2] = E[2];
    EH_out[o + 3] = H[0];
    EH_out[o + 4] = H[1];
    EH_out[o + 5] = H[2];
}
