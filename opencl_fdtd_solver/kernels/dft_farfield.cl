/* Copyright (C) 2026: OpenCL FDTD Solver Contributors
 * Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
 *
 * This file is part of opencl-fdtd-solver (GPLv3 or later).
 */

inline void dft_add(__global accreal2 *slot, real val, accreal pr, accreal pi) {
    accreal2 cur = *slot;
    cur.x += (accreal)val * pr;
    cur.y += (accreal)val * pi;
    *slot = cur;
}

/*
 * One launch over packed face samples of a closed Yee Huygens box
 * [ix0..ix1] x [iy0..iy1] x [iz0..iz1]:
 *   x-faces (i = ix0, ix1): nyf = iy1 - iy0, nzf = iz1 - iz0 cells at (i, j+0.5, k+0.5)
 *   y-faces (j = iy0, iy1): nxf = ix1 - ix0, nzf = iz1 - iz0 cells at (i+0.5, j, k+0.5)
 *   z-faces (k = iz0, iz1): nxf = ix1 - ix0, nyf = iy1 - iy0 cells at (i+0.5, j+0.5, k)
 *
 * Phase args and output buffers accumulate at `accreal` precision (double when
 * ACCUM_FP64=1) even when volumetric fields (`Ex`..`Hz`) are `real` (fp32).
 */
__kernel void accumulate_dft_faces_fused(
    int Nx, int Ny, int Nz,
    int ix0, int ix1,
    int iy0, int iy1,
    int iz0, int iz1,
    int off0, int off1, int off2, int off3, int off4, int off5,
    int n_face,
    accreal phase_e_real, accreal phase_e_imag,
    accreal phase_h_real, accreal phase_h_imag,
    __global const real * restrict Ex,
    __global const real * restrict Ey,
    __global const real * restrict Ez,
    __global const real * restrict Hx,
    __global const real * restrict Hy,
    __global const real * restrict Hz,
    __global accreal2 * restrict Ex_dft,
    __global accreal2 * restrict Ey_dft,
    __global accreal2 * restrict Ez_dft,
    __global accreal2 * restrict Hx_dft,
    __global accreal2 * restrict Hy_dft,
    __global accreal2 * restrict Hz_dft
) {
    int face_i = get_global_id(0);
    if (face_i >= n_face) return;

    int nxf = ix1 - ix0;
    int nyf = iy1 - iy0;
    int nzf = iz1 - iz0;
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

    accreal pr_e = phase_e_real, pi_e = phase_e_imag;
    accreal pr_h = phase_h_real, pi_h = phase_h_imag;

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
        real hx01 = AT(Hx, i1, abs_j, abs_k), hx11 = AT(Hx, i1, abs_j, k_lo);
        real hy00 = AT(Hy, abs_i, abs_j, abs_k), hy10 = AT(Hy, abs_i, abs_j, k_lo);
        real hy01 = AT(Hy, abs_i, j1, abs_k), hy11 = AT(Hy, abs_i, j1, k_lo);
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
    __global const accreal2 * restrict Ex_c,
    __global const accreal2 * restrict Ey_c,
    __global const accreal2 * restrict Ez_c,
    __global const accreal2 * restrict Hx_c,
    __global const accreal2 * restrict Hy_c,
    __global const accreal2 * restrict Hz_c,
    __global const accreal2 * restrict Ex_p,
    __global const accreal2 * restrict Ey_p,
    __global const accreal2 * restrict Ez_p,
    __global const accreal2 * restrict Hx_p,
    __global const accreal2 * restrict Hy_p,
    __global const accreal2 * restrict Hz_p,
    __global accreal * restrict partial_num,
    __global accreal * restrict partial_den,
    __local accreal *sn,
    __local accreal *sd
) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int grp = get_group_id(0);
    accreal nsum = (accreal)0.0;
    accreal dsum = (accreal)0.0;
    if (gid < n) {
        accreal2 c, p, d;
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

/* accreal2 counterparts — used for the N/L reduction (see face_sample_NL /
 * farfield_accumulate_nl below), which sums ~1e6 face samples per
 * observation angle and benefits from the same accreal precision as the
 * DFT accumulator it reads from. */
inline accreal2 cmul_acc(accreal2 a, accreal2 b) {
    return (accreal2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline void caccum_acc(accreal2 *acc, accreal2 val) {
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

/*
 * accreal variants: the N/L near-to-far reduction sums ~1e6 face samples
 * per observation angle via atomic add — at least as prone to round-off
 * compounding as the per-step DFT accumulation above, so it gets the same
 * accreal (double, by default) treatment independent of `real`.
 */
inline void atomic_add_accreal(__global volatile accreal *addr, accreal val) {
#if ACCUM_FP64
    union {
        ulong u;
        double f;
    } oldv, newv;
    do {
        oldv.f = *addr;
        newv.f = oldv.f + val;
    } while (atom_cmpxchg((__global volatile ulong *)addr, oldv.u, newv.u) != oldv.u);
#else
    atomic_add_real((__global volatile real *)addr, (real)val);
#endif
}

inline void atomic_add_accreal2(__global accreal2 *addr, accreal2 val) {
    __global accreal *p = (__global accreal *)addr;
    atomic_add_accreal(&p[0], val.x);
    atomic_add_accreal(&p[1], val.y);
}

/* Contribution of one closed-box midpoint face sample into N,L (6 accreal2).
 * All coordinates, phase, and quadrature weights run in accreal precision. */
inline void face_sample_NL(
    int face_i,
    accreal rx, accreal ry, accreal rz,
    accreal k_wave, accreal dA, accreal dl,
    int ix0, int ix1, int iy0, int iy1, int iz0, int iz1,
    int off0, int off1, int off2, int off3, int off4, int off5,
    __global const accreal2 *Ex_f,
    __global const accreal2 *Ey_f,
    __global const accreal2 *Ez_f,
    __global const accreal2 *Hx_f,
    __global const accreal2 *Hy_f,
    __global const accreal2 *Hz_f,
    accreal2 *Nx, accreal2 *Ny, accreal2 *Nz,
    accreal2 *Lx, accreal2 *Ly, accreal2 *Lz
) {
    int nxf = ix1 - ix0;
    int nyf = iy1 - iy0;
    int nzf = iz1 - iz0;
    int face, loc, abs_i, abs_j, abs_k;
    accreal nfa, xp, yp, zp;

    if (face_i < off1) {
        face = 0; loc = face_i - off0; nfa = -(accreal)1.0;
        abs_i = ix0; abs_j = iy0 + loc / nzf; abs_k = iz0 + (loc - (loc / nzf) * nzf);
    } else if (face_i < off2) {
        face = 1; loc = face_i - off1; nfa = (accreal)1.0;
        abs_i = ix1; abs_j = iy0 + loc / nzf; abs_k = iz0 + (loc - (loc / nzf) * nzf);
    } else if (face_i < off3) {
        face = 2; loc = face_i - off2; nfa = -(accreal)1.0;
        abs_i = ix0 + loc / nzf; abs_j = iy0; abs_k = iz0 + (loc - (loc / nzf) * nzf);
    } else if (face_i < off4) {
        face = 3; loc = face_i - off3; nfa = (accreal)1.0;
        abs_i = ix0 + loc / nzf; abs_j = iy1; abs_k = iz0 + (loc - (loc / nzf) * nzf);
    } else if (face_i < off5) {
        face = 4; loc = face_i - off4; nfa = -(accreal)1.0;
        abs_i = ix0 + loc / nyf; abs_j = iy0 + (loc - (loc / nyf) * nyf); abs_k = iz0;
    } else {
        face = 5; loc = face_i - off5; nfa = (accreal)1.0;
        abs_i = ix0 + loc / nyf; abs_j = iy0 + (loc - (loc / nyf) * nyf); abs_k = iz1;
    }
    (void)nxf;
    /* Closed-box midpoint quadrature: each sample owns an exact dl x dl patch
     * centered at the face-center with uniform weight 1.0 (no gaps or overhang). */
    if (face <= 1) {
        xp = (accreal)abs_i * dl;
        yp = ((accreal)abs_j + (accreal)0.5) * dl;
        zp = ((accreal)abs_k + (accreal)0.5) * dl;
    } else if (face <= 3) {
        xp = ((accreal)abs_i + (accreal)0.5) * dl;
        yp = (accreal)abs_j * dl;
        zp = ((accreal)abs_k + (accreal)0.5) * dl;
    } else {
        xp = ((accreal)abs_i + (accreal)0.5) * dl;
        yp = ((accreal)abs_j + (accreal)0.5) * dl;
        zp = (accreal)abs_k * dl;
    }
    int li = face_i;
    accreal phase = k_wave * (rx * xp + ry * yp + rz * zp);
    accreal2 ph = (accreal2)(cos(phase), -sin(phase));
    accreal dAw = dA;

    if (face <= 1) {
        accreal2 Jy = cmul_acc((accreal2)( nfa * Hz_f[li].x,  nfa * Hz_f[li].y), ph);
        accreal2 Jz = cmul_acc((accreal2)(-nfa * Hy_f[li].x, -nfa * Hy_f[li].y), ph);
        accreal2 My = cmul_acc((accreal2)(-nfa * Ez_f[li].x, -nfa * Ez_f[li].y), ph);
        accreal2 Mz = cmul_acc((accreal2)( nfa * Ey_f[li].x,  nfa * Ey_f[li].y), ph);
        caccum_acc(Ny, (accreal2)(Jy.x * dAw, Jy.y * dAw));
        caccum_acc(Nz, (accreal2)(Jz.x * dAw, Jz.y * dAw));
        caccum_acc(Ly, (accreal2)(My.x * dAw, My.y * dAw));
        caccum_acc(Lz, (accreal2)(Mz.x * dAw, Mz.y * dAw));
    } else if (face <= 3) {
        accreal2 Jx = cmul_acc((accreal2)(-nfa * Hz_f[li].x, -nfa * Hz_f[li].y), ph);
        accreal2 Jz = cmul_acc((accreal2)( nfa * Hx_f[li].x,  nfa * Hx_f[li].y), ph);
        accreal2 Mx = cmul_acc((accreal2)( nfa * Ez_f[li].x,  nfa * Ez_f[li].y), ph);
        accreal2 Mz = cmul_acc((accreal2)(-nfa * Ex_f[li].x, -nfa * Ex_f[li].y), ph);
        caccum_acc(Nx, (accreal2)(Jx.x * dAw, Jx.y * dAw));
        caccum_acc(Nz, (accreal2)(Jz.x * dAw, Jz.y * dAw));
        caccum_acc(Lx, (accreal2)(Mx.x * dAw, Mx.y * dAw));
        caccum_acc(Lz, (accreal2)(Mz.x * dAw, Mz.y * dAw));
    } else {
        accreal2 Jx = cmul_acc((accreal2)( nfa * Hy_f[li].x,  nfa * Hy_f[li].y), ph);
        accreal2 Jy = cmul_acc((accreal2)(-nfa * Hx_f[li].x, -nfa * Hx_f[li].y), ph);
        accreal2 Mx = cmul_acc((accreal2)(-nfa * Ey_f[li].x, -nfa * Ey_f[li].y), ph);
        accreal2 My = cmul_acc((accreal2)( nfa * Ex_f[li].x,  nfa * Ex_f[li].y), ph);
        caccum_acc(Nx, (accreal2)(Jx.x * dAw, Jx.y * dAw));
        caccum_acc(Ny, (accreal2)(Jy.x * dAw, Jy.y * dAw));
        caccum_acc(Lx, (accreal2)(Mx.x * dAw, Mx.y * dAw));
        caccum_acc(Ly, (accreal2)(My.x * dAw, My.y * dAw));
    }
}

/*
 * Parallel over face samples (dim0) and observation points (dim1).
 * Local reduction along dim0, then atomic add into NL[obs*6 + c].
 */
__kernel void farfield_accumulate_nl(
    int n_face,
    int n_obs,
    __global const accreal *obs_xyz,
    accreal k_wave,
    accreal dl,
    int ix0, int ix1,
    int iy0, int iy1,
    int iz0, int iz1,
    int off0, int off1, int off2, int off3, int off4, int off5,
    __global const accreal2 *Ex_f,
    __global const accreal2 *Ey_f,
    __global const accreal2 *Ez_f,
    __global const accreal2 *Hx_f,
    __global const accreal2 *Hy_f,
    __global const accreal2 *Hz_f,
    __global accreal2 *NL_out,
    __local accreal2 *scratch
) {
    int face_i = get_global_id(0);
    int obs = get_global_id(1);
    int lid = get_local_id(0);
    int lsize = get_local_size(0);

    accreal2 Nx = (accreal2)((accreal)0.0, (accreal)0.0);
    accreal2 Ny = (accreal2)((accreal)0.0, (accreal)0.0);
    accreal2 Nz = (accreal2)((accreal)0.0, (accreal)0.0);
    accreal2 Lx = (accreal2)((accreal)0.0, (accreal)0.0);
    accreal2 Ly = (accreal2)((accreal)0.0, (accreal)0.0);
    accreal2 Lz = (accreal2)((accreal)0.0, (accreal)0.0);

    if (face_i < n_face && obs < n_obs) {
        accreal ox = obs_xyz[3 * obs + 0];
        accreal oy = obs_xyz[3 * obs + 1];
        accreal oz = obs_xyz[3 * obs + 2];
        accreal r = sqrt(ox * ox + oy * oy + oz * oz);
        if (r < (accreal)1.0e-30) r = (accreal)1.0e-30;
        accreal rx = ox / r, ry = oy / r, rz = oz / r;
        accreal dA = dl * dl;
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
        atomic_add_accreal2(&NL_out[base + 0], scratch[0 * lsize]);
        atomic_add_accreal2(&NL_out[base + 1], scratch[1 * lsize]);
        atomic_add_accreal2(&NL_out[base + 2], scratch[2 * lsize]);
        atomic_add_accreal2(&NL_out[base + 3], scratch[3 * lsize]);
        atomic_add_accreal2(&NL_out[base + 4], scratch[4 * lsize]);
        atomic_add_accreal2(&NL_out[base + 5], scratch[5 * lsize]);
    }
}

/* Convert integrated N,L → far E,H for each observation. */
__kernel void farfield_nl_to_eh(
    int n_obs,
    __global const accreal *obs_xyz,
    accreal k_wave,
    accreal eta0,
    __global const accreal2 *NL_in,
    __global accreal2 *EH_out
) {
    int p = get_global_id(0);
    if (p >= n_obs) return;

    accreal ox = obs_xyz[3 * p + 0];
    accreal oy = obs_xyz[3 * p + 1];
    accreal oz = obs_xyz[3 * p + 2];
    accreal r = sqrt(ox * ox + oy * oy + oz * oz);
    if (r < (accreal)1.0e-30) r = (accreal)1.0e-30;
    accreal rx = ox / r, ry = oy / r, rz = oz / r;
    accreal rhat[3] = { rx, ry, rz };

    accreal2 Nvec[3] = { NL_in[6 * p + 0], NL_in[6 * p + 1], NL_in[6 * p + 2] };
    accreal2 Lvec[3] = { NL_in[6 * p + 3], NL_in[6 * p + 4], NL_in[6 * p + 5] };

    accreal k_wave_a = (accreal)k_wave, eta0_a = (accreal)eta0;
    accreal ang = k_wave_a * r;
    // Positive-time DFT: outgoing wave ~ exp(+i k r).
    accreal2 eikr = (accreal2)(cos(ang), sin(ang));
    accreal scale = k_wave_a / ((accreal)4.0 * (accreal)3.14159265358979323846 * r);
    accreal2 pref = cmul_acc((accreal2)((accreal)0.0, -scale), eikr);

    accreal2 rxL[3], Nt[3], E[3], H[3];
    rxL[0] = (accreal2)(rhat[1] * Lvec[2].x - rhat[2] * Lvec[1].x,
                         rhat[1] * Lvec[2].y - rhat[2] * Lvec[1].y);
    rxL[1] = (accreal2)(rhat[2] * Lvec[0].x - rhat[0] * Lvec[2].x,
                         rhat[2] * Lvec[0].y - rhat[0] * Lvec[2].y);
    rxL[2] = (accreal2)(rhat[0] * Lvec[1].x - rhat[1] * Lvec[0].x,
                         rhat[0] * Lvec[1].y - rhat[1] * Lvec[0].y);

    accreal2 Ndot = (accreal2)(
        rhat[0] * Nvec[0].x + rhat[1] * Nvec[1].x + rhat[2] * Nvec[2].x,
        rhat[0] * Nvec[0].y + rhat[1] * Nvec[1].y + rhat[2] * Nvec[2].y);

    for (int c = 0; c < 3; c++) {
        Nt[c] = (accreal2)(Nvec[c].x - Ndot.x * rhat[c], Nvec[c].y - Ndot.y * rhat[c]);
        /* Face integrals store H x n and n x E (the negatives of the
         * conventional equivalent currents). With the +i*omega*t DFT:
         * E = -ik exp(+ikr)/(4 pi r) * [eta*Nt - r_hat x L]. */
        accreal2 tE = (accreal2)(eta0_a * Nt[c].x - rxL[c].x, eta0_a * Nt[c].y - rxL[c].y);
        E[c] = cmul_acc(pref, tE);
    }
    // Outgoing TEM: H = r̂ × E / η, giving positive radial energy flux.
    H[0] = (accreal2)((rhat[1] * E[2].x - rhat[2] * E[1].x) / eta0_a,
                       (rhat[1] * E[2].y - rhat[2] * E[1].y) / eta0_a);
    H[1] = (accreal2)((rhat[2] * E[0].x - rhat[0] * E[2].x) / eta0_a,
                       (rhat[2] * E[0].y - rhat[0] * E[2].y) / eta0_a);
    H[2] = (accreal2)((rhat[0] * E[1].x - rhat[1] * E[0].x) / eta0_a,
                       (rhat[0] * E[1].y - rhat[1] * E[0].y) / eta0_a);

    int o = 6 * p;
    EH_out[o + 0] = E[0];
    EH_out[o + 1] = E[1];
    EH_out[o + 2] = E[2];
    EH_out[o + 3] = H[0];
    EH_out[o + 4] = H[1];
    EH_out[o + 5] = H[2];
}
