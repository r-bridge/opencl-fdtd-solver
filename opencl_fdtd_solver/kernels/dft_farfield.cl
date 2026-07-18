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
    real phase_real, real phase_imag,
    __global const real *field,
    __global real2 *field_dft
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
        real val = field[idx];

        real2 current_dft = field_dft[idx];
        current_dft.x += val * phase_real;
        current_dft.y += val * phase_imag;
        field_dft[idx] = current_dft;
    }
}

/*
 * Accumulate DFT onto one Huygens face into a packed real2 buffer.
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
    real phase_real, real phase_imag,
    __global const real *field,
    __global real2 *face_dft
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
    real val = field[idx];
    int dst = face_offset + face_li;
    real2 cur = face_dft[dst];
    cur.x += val * phase_real;
    cur.y += val * phase_imag;
    face_dft[dst] = cur;
}

inline void dft_add(__global real2 *slot, real val, real pr, real pi) {
    real2 cur = *slot;
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
    real phase_e_real, real phase_e_imag,
    real phase_h_real, real phase_h_imag,
    __global const real * restrict Ex,
    __global const real * restrict Ey,
    __global const real * restrict Ez,
    __global const real * restrict Hx,
    __global const real * restrict Hy,
    __global const real * restrict Hz,
    __global real2 * restrict Ex_dft,
    __global real2 * restrict Ey_dft,
    __global real2 * restrict Ez_dft,
    __global real2 * restrict Hx_dft,
    __global real2 * restrict Hy_dft,
    __global real2 * restrict Hz_dft
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
    real pr_e = phase_e_real, pi_e = phase_e_imag;
    real pr_h = phase_h_real, pi_h = phase_h_imag;

/* Co-locate tangential E/H at the face-center (half-cell averages), then DFT.
 * H uses phase_h = phase_e * exp(-j ω Δt/2): at monitor time E is at
 * integer t while H is still at (n+1/2)Δt after the leapfrog update. */
#define AT(F, ii, jj, kk) ((F)[((ii) * Ny + (jj)) * Nz + (kk)])
#define AVG2(a, b, ok) ((ok) ? (real)0.5 * ((a) + (b)) : (a))
    if (face <= 1) {
        /* Face center (i, j+½, k+½): avg Ey in z, Ez in y; H from both sides of face. */
        int k1 = (abs_k + 1 < Nz) ? abs_k + 1 : abs_k;
        int j1 = (abs_j + 1 < Ny) ? abs_j + 1 : abs_j;
        int i_lo = (abs_i > 0) ? abs_i - 1 : abs_i;
        real ey = AVG2(AT(Ey, abs_i, abs_j, abs_k), AT(Ey, abs_i, abs_j, k1), k1 != abs_k);
        real ez = AVG2(AT(Ez, abs_i, abs_j, abs_k), AT(Ez, abs_i, j1, abs_k), j1 != abs_j);
        real hy00 = AT(Hy, abs_i, abs_j, abs_k), hy10 = AT(Hy, i_lo, abs_j, abs_k);
        real hy01 = AT(Hy, abs_i, j1, abs_k), hy11 = AT(Hy, i_lo, j1, abs_k);
        real hz00 = AT(Hz, abs_i, abs_j, abs_k), hz10 = AT(Hz, i_lo, abs_j, abs_k);
        real hz01 = AT(Hz, abs_i, abs_j, k1), hz11 = AT(Hz, i_lo, abs_j, k1);
        real hy = (real)0.25 * (hy00 + hy10 + hy01 + hy11);
        real hz = (real)0.25 * (hz00 + hz10 + hz01 + hz11);
        dft_add(&Ey_dft[face_i], ey, pr_e, pi_e);
        dft_add(&Ez_dft[face_i], ez, pr_e, pi_e);
        dft_add(&Hy_dft[face_i], hy, pr_h, pi_h);
        dft_add(&Hz_dft[face_i], hz, pr_h, pi_h);
    } else if (face <= 3) {
        /* Face center (i+½, j, k+½). */
        int i1 = (abs_i + 1 < Nx) ? abs_i + 1 : abs_i;
        int k1 = (abs_k + 1 < Nz) ? abs_k + 1 : abs_k;
        int j_lo = (abs_j > 0) ? abs_j - 1 : abs_j;
        real ex = AVG2(AT(Ex, abs_i, abs_j, abs_k), AT(Ex, abs_i, abs_j, k1), k1 != abs_k);
        real ez = AVG2(AT(Ez, abs_i, abs_j, abs_k), AT(Ez, i1, abs_j, abs_k), i1 != abs_i);
        real hx00 = AT(Hx, abs_i, abs_j, abs_k), hx10 = AT(Hx, abs_i, j_lo, abs_k);
        real hx01 = AT(Hx, i1, abs_j, abs_k), hx11 = AT(Hx, i1, j_lo, abs_k);
        real hz00 = AT(Hz, abs_i, abs_j, abs_k), hz10 = AT(Hz, abs_i, j_lo, abs_k);
        real hz01 = AT(Hz, abs_i, abs_j, k1), hz11 = AT(Hz, abs_i, j_lo, k1);
        real hx = (real)0.25 * (hx00 + hx10 + hx01 + hx11);
        real hz = (real)0.25 * (hz00 + hz10 + hz01 + hz11);
        dft_add(&Ex_dft[face_i], ex, pr_e, pi_e);
        dft_add(&Ez_dft[face_i], ez, pr_e, pi_e);
        dft_add(&Hx_dft[face_i], hx, pr_h, pi_h);
        dft_add(&Hz_dft[face_i], hz, pr_h, pi_h);
    } else {
        /* Face center (i+½, j+½, k). */
        int i1 = (abs_i + 1 < Nx) ? abs_i + 1 : abs_i;
        int j1 = (abs_j + 1 < Ny) ? abs_j + 1 : abs_j;
        int k_lo = (abs_k > 0) ? abs_k - 1 : abs_k;
        real ex = AVG2(AT(Ex, abs_i, abs_j, abs_k), AT(Ex, abs_i, j1, abs_k), j1 != abs_j);
        real ey = AVG2(AT(Ey, abs_i, abs_j, abs_k), AT(Ey, i1, abs_j, abs_k), i1 != abs_i);
        real hx00 = AT(Hx, abs_i, abs_j, abs_k), hx10 = AT(Hx, abs_i, abs_j, k_lo);
        real hx01 = AT(Hx, abs_i, j1, abs_k), hx11 = AT(Hx, abs_i, j1, k_lo);
        real hy00 = AT(Hy, abs_i, abs_j, abs_k), hy10 = AT(Hy, abs_i, abs_j, k_lo);
        real hy01 = AT(Hy, i1, abs_j, abs_k), hy11 = AT(Hy, i1, abs_j, k_lo);
        real hx = (real)0.25 * (hx00 + hx10 + hx01 + hx11);
        real hy = (real)0.25 * (hy00 + hy10 + hy01 + hy11);
        dft_add(&Ex_dft[face_i], ex, pr_e, pi_e);
        dft_add(&Ey_dft[face_i], ey, pr_e, pi_e);
        dft_add(&Hx_dft[face_i], hx, pr_h, pi_h);
        dft_add(&Hy_dft[face_i], hy, pr_h, pi_h);
    }
#undef AT
#undef AVG2
}

/*
 * Relative L2 change of face DFT vs a previous snapshot:
 *   sqrt(sum |cur-prev|^2 / sum |cur|^2)
 * Workgroup-local reduction; host sums partials (tiny download).
 */
__kernel void dft_rel_change_partial(
    int n,
    __global const real2 * restrict Ex_c,
    __global const real2 * restrict Ey_c,
    __global const real2 * restrict Ez_c,
    __global const real2 * restrict Hx_c,
    __global const real2 * restrict Hy_c,
    __global const real2 * restrict Hz_c,
    __global const real2 * restrict Ex_p,
    __global const real2 * restrict Ey_p,
    __global const real2 * restrict Ez_p,
    __global const real2 * restrict Hx_p,
    __global const real2 * restrict Hy_p,
    __global const real2 * restrict Hz_p,
    __global real * restrict partial_num,
    __global real * restrict partial_den,
    __local real *sn,
    __local real *sd
) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int grp = get_group_id(0);
    real nsum = (real)0.0;
    real dsum = (real)0.0;
    if (gid < n) {
        real2 c, p, d;
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

inline real2 cmul(real2 a, real2 b) {
    return (real2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline void caccum(real2 *acc, real2 val) {
    acc->x += val.x;
    acc->y += val.y;
}

inline void atomic_add_real(__global volatile real *addr, real val) {
#if USE_FP64
    /* 64-bit CAS; needs cl_khr_int64_base_atomics (enabled in precision.cl). */
    union {
        ulong u;
        double f;
    } oldv, newv;
    do {
        oldv.f = *addr;
        newv.f = oldv.f + val;
    } while (atom_cmpxchg((__global volatile ulong *)addr, oldv.u, newv.u) != oldv.u);
#else
    union {
        unsigned int u;
        float f;
    } oldv, newv;
    do {
        oldv.f = *addr;
        newv.f = oldv.f + val;
    } while (atomic_cmpxchg((__global volatile unsigned int *)addr, oldv.u, newv.u)
             != oldv.u);
#endif
}

inline void atomic_add_real2(__global real2 *addr, real2 val) {
    __global real *p = (__global real *)addr;
    atomic_add_real(&p[0], val.x);
    atomic_add_real(&p[1], val.y);
}

/* Contribution of one packed face sample into N,L (6 real2). */
inline void face_sample_NL(
    int face_i,
    real rx, real ry, real rz,
    real k_wave, real dA, real dl,
    int ix0, int ix1, int iy0, int iy1, int iz0, int iz1,
    int off0, int off1, int off2, int off3, int off4, int off5,
    __global const real2 *Ex_f,
    __global const real2 *Ey_f,
    __global const real2 *Ez_f,
    __global const real2 *Hx_f,
    __global const real2 *Hy_f,
    __global const real2 *Hz_f,
    real2 *Nx, real2 *Ny, real2 *Nz,
    real2 *Lx, real2 *Ly, real2 *Lz
) {
    int nxf = ix1 - ix0 + 1;
    int nyf = iy1 - iy0 + 1;
    int nzf = iz1 - iz0 + 1;
    int face, loc, abs_i, abs_j, abs_k;
    real nf, xp, yp, zp;

    if (face_i < off1) {
        face = 0; loc = face_i - off0; nf = -(real)1.0;
        abs_i = ix0; abs_j = iy0 + loc / nzf; abs_k = iz0 + (loc - (loc / nzf) * nzf);
    } else if (face_i < off2) {
        face = 1; loc = face_i - off1; nf = (real)1.0;
        abs_i = ix1; abs_j = iy0 + loc / nzf; abs_k = iz0 + (loc - (loc / nzf) * nzf);
    } else if (face_i < off3) {
        face = 2; loc = face_i - off2; nf = -(real)1.0;
        abs_i = ix0 + loc / nzf; abs_j = iy0; abs_k = iz0 + (loc - (loc / nzf) * nzf);
    } else if (face_i < off4) {
        face = 3; loc = face_i - off3; nf = (real)1.0;
        abs_i = ix0 + loc / nzf; abs_j = iy1; abs_k = iz0 + (loc - (loc / nzf) * nzf);
    } else if (face_i < off5) {
        face = 4; loc = face_i - off4; nf = -(real)1.0;
        abs_i = ix0 + loc / nyf; abs_j = iy0 + (loc - (loc / nyf) * nyf); abs_k = iz0;
    } else {
        face = 5; loc = face_i - off5; nf = (real)1.0;
        abs_i = ix0 + loc / nyf; abs_j = iy0 + (loc - (loc / nyf) * nyf); abs_k = iz1;
    }
    (void)nxf;
    /* Surface sample at face-center (half-cell offset in the two tangential axes). */
    if (face <= 1) {
        xp = abs_i * dl;
        yp = (abs_j + (real)0.5) * dl;
        zp = (abs_k + (real)0.5) * dl;
    } else if (face <= 3) {
        xp = (abs_i + (real)0.5) * dl;
        yp = abs_j * dl;
        zp = (abs_k + (real)0.5) * dl;
    } else {
        xp = (abs_i + (real)0.5) * dl;
        yp = (abs_j + (real)0.5) * dl;
        zp = abs_k * dl;
    }
    int li = face_i;
    real phase = k_wave * (rx * xp + ry * yp + rz * zp);
    real2 ph = (real2)(cos(phase), sin(phase));

    /* Trapezoidal face quadrature: half weight on edges, quarter on corners. */
    real wj, wk;
    if (face <= 1) {
        wj = (abs_j == iy0 || abs_j == iy1) ? (real)0.5 : (real)1.0;
        wk = (abs_k == iz0 || abs_k == iz1) ? (real)0.5 : (real)1.0;
    } else if (face <= 3) {
        wj = (abs_i == ix0 || abs_i == ix1) ? (real)0.5 : (real)1.0;
        wk = (abs_k == iz0 || abs_k == iz1) ? (real)0.5 : (real)1.0;
    } else {
        wj = (abs_i == ix0 || abs_i == ix1) ? (real)0.5 : (real)1.0;
        wk = (abs_j == iy0 || abs_j == iy1) ? (real)0.5 : (real)1.0;
    }
    real dAw = dA * wj * wk;

    if (face <= 1) {
        real2 Jy = cmul((real2)( nf * Hz_f[li].x,  nf * Hz_f[li].y), ph);
        real2 Jz = cmul((real2)(-nf * Hy_f[li].x, -nf * Hy_f[li].y), ph);
        real2 My = cmul((real2)(-nf * Ez_f[li].x, -nf * Ez_f[li].y), ph);
        real2 Mz = cmul((real2)( nf * Ey_f[li].x,  nf * Ey_f[li].y), ph);
        caccum(Ny, (real2)(Jy.x * dAw, Jy.y * dAw));
        caccum(Nz, (real2)(Jz.x * dAw, Jz.y * dAw));
        caccum(Ly, (real2)(My.x * dAw, My.y * dAw));
        caccum(Lz, (real2)(Mz.x * dAw, Mz.y * dAw));
    } else if (face <= 3) {
        real2 Jx = cmul((real2)(-nf * Hz_f[li].x, -nf * Hz_f[li].y), ph);
        real2 Jz = cmul((real2)( nf * Hx_f[li].x,  nf * Hx_f[li].y), ph);
        real2 Mx = cmul((real2)( nf * Ez_f[li].x,  nf * Ez_f[li].y), ph);
        real2 Mz = cmul((real2)(-nf * Ex_f[li].x, -nf * Ex_f[li].y), ph);
        caccum(Nx, (real2)(Jx.x * dAw, Jx.y * dAw));
        caccum(Nz, (real2)(Jz.x * dAw, Jz.y * dAw));
        caccum(Lx, (real2)(Mx.x * dAw, Mx.y * dAw));
        caccum(Lz, (real2)(Mz.x * dAw, Mz.y * dAw));
    } else {
        real2 Jx = cmul((real2)( nf * Hy_f[li].x,  nf * Hy_f[li].y), ph);
        real2 Jy = cmul((real2)(-nf * Hx_f[li].x, -nf * Hx_f[li].y), ph);
        real2 Mx = cmul((real2)(-nf * Ey_f[li].x, -nf * Ey_f[li].y), ph);
        real2 My = cmul((real2)( nf * Ex_f[li].x,  nf * Ex_f[li].y), ph);
        caccum(Nx, (real2)(Jx.x * dAw, Jx.y * dAw));
        caccum(Ny, (real2)(Jy.x * dAw, Jy.y * dAw));
        caccum(Lx, (real2)(Mx.x * dAw, Mx.y * dAw));
        caccum(Ly, (real2)(My.x * dAw, My.y * dAw));
    }
}

/*
 * Parallel over face samples (dim0) and observation points (dim1).
 * Local reduction along dim0, then atomic add into NL[obs*6 + c].
 */
__kernel void farfield_accumulate_nl(
    int n_face,
    int n_obs,
    __global const real *obs_xyz,
    real k_wave,
    real dl,
    int ix0, int ix1,
    int iy0, int iy1,
    int iz0, int iz1,
    int off0, int off1, int off2, int off3, int off4, int off5,
    __global const real2 *Ex_f,
    __global const real2 *Ey_f,
    __global const real2 *Ez_f,
    __global const real2 *Hx_f,
    __global const real2 *Hy_f,
    __global const real2 *Hz_f,
    __global real2 *NL_out,
    __local real2 *scratch
) {
    int face_i = get_global_id(0);
    int obs = get_global_id(1);
    int lid = get_local_id(0);
    int lsize = get_local_size(0);

    real2 Nx = (real2)((real)0.0, (real)0.0);
    real2 Ny = (real2)((real)0.0, (real)0.0);
    real2 Nz = (real2)((real)0.0, (real)0.0);
    real2 Lx = (real2)((real)0.0, (real)0.0);
    real2 Ly = (real2)((real)0.0, (real)0.0);
    real2 Lz = (real2)((real)0.0, (real)0.0);

    if (face_i < n_face && obs < n_obs) {
        real ox = obs_xyz[3 * obs + 0];
        real oy = obs_xyz[3 * obs + 1];
        real oz = obs_xyz[3 * obs + 2];
        real r = sqrt(ox * ox + oy * oy + oz * oz);
        if (r < (real)1.0e-30) r = (real)1.0e-30;
        real rx = ox / r, ry = oy / r, rz = oz / r;
        real dA = dl * dl;
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
        atomic_add_real2(&NL_out[base + 0], scratch[0 * lsize]);
        atomic_add_real2(&NL_out[base + 1], scratch[1 * lsize]);
        atomic_add_real2(&NL_out[base + 2], scratch[2 * lsize]);
        atomic_add_real2(&NL_out[base + 3], scratch[3 * lsize]);
        atomic_add_real2(&NL_out[base + 4], scratch[4 * lsize]);
        atomic_add_real2(&NL_out[base + 5], scratch[5 * lsize]);
    }
}

/* Convert integrated N,L → far E,H for each observation. */
__kernel void farfield_nl_to_eh(
    int n_obs,
    __global const real *obs_xyz,
    real k_wave,
    real eta0,
    __global const real2 *NL_in,
    __global real2 *EH_out
) {
    int p = get_global_id(0);
    if (p >= n_obs) return;

    real ox = obs_xyz[3 * p + 0];
    real oy = obs_xyz[3 * p + 1];
    real oz = obs_xyz[3 * p + 2];
    real r = sqrt(ox * ox + oy * oy + oz * oz);
    if (r < (real)1.0e-30) r = (real)1.0e-30;
    real rx = ox / r, ry = oy / r, rz = oz / r;
    real rhat[3] = { rx, ry, rz };

    real2 Nvec[3] = { NL_in[6 * p + 0], NL_in[6 * p + 1], NL_in[6 * p + 2] };
    real2 Lvec[3] = { NL_in[6 * p + 3], NL_in[6 * p + 4], NL_in[6 * p + 5] };

    real ang = k_wave * r;
    // Outgoing wave ~ e^{-j k r}
    real2 eikr = (real2)(cos(ang), -sin(ang));
    real scale = k_wave / ((real)4.0 * (real)3.14159265358979323846 * r);
    real2 pref = cmul((real2)((real)0.0, -scale), eikr);

    real2 rxL[3], Nt[3], E[3], H[3];
    rxL[0] = (real2)(rhat[1] * Lvec[2].x - rhat[2] * Lvec[1].x,
                      rhat[1] * Lvec[2].y - rhat[2] * Lvec[1].y);
    rxL[1] = (real2)(rhat[2] * Lvec[0].x - rhat[0] * Lvec[2].x,
                      rhat[2] * Lvec[0].y - rhat[0] * Lvec[2].y);
    rxL[2] = (real2)(rhat[0] * Lvec[1].x - rhat[1] * Lvec[0].x,
                      rhat[0] * Lvec[1].y - rhat[1] * Lvec[0].y);

    real2 Ndot = (real2)(
        rhat[0] * Nvec[0].x + rhat[1] * Nvec[1].x + rhat[2] * Nvec[2].x,
        rhat[0] * Nvec[0].y + rhat[1] * Nvec[1].y + rhat[2] * Nvec[2].y);

    for (int c = 0; c < 3; c++) {
        Nt[c] = (real2)(Nvec[c].x - Ndot.x * rhat[c], Nvec[c].y - Ndot.y * rhat[c]);
        real2 tE = (real2)(eta0 * Nt[c].x + rxL[c].x, eta0 * Nt[c].y + rxL[c].y);
        E[c] = cmul(pref, tE);
    }
    // Far-field TEM: H = -r̂ × E / η
    H[0] = (real2)(-(rhat[1] * E[2].x - rhat[2] * E[1].x) / eta0,
                    -(rhat[1] * E[2].y - rhat[2] * E[1].y) / eta0);
    H[1] = (real2)(-(rhat[2] * E[0].x - rhat[0] * E[2].x) / eta0,
                    -(rhat[2] * E[0].y - rhat[0] * E[2].y) / eta0);
    H[2] = (real2)(-(rhat[0] * E[1].x - rhat[1] * E[0].x) / eta0,
                    -(rhat[0] * E[1].y - rhat[1] * E[0].y) / eta0);

    int o = 6 * p;
    EH_out[o + 0] = E[0];
    EH_out[o + 1] = E[1];
    EH_out[o + 2] = E[2];
    EH_out[o + 3] = H[0];
    EH_out[o + 4] = H[1];
    EH_out[o + 5] = H[2];
}
