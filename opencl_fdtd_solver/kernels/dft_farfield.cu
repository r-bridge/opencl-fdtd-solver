/* Copyright (C) 2026: OpenCL FDTD Solver Contributors
 * Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
 *
 * This file is part of opencl-fdtd-solver (GPLv3 or later).
 *
 * CUDA port of dft_farfield.cl (see yee_update.cu for the real/real2 defines).
 */

/* Legacy full-box DFT (volume buffer); prefer accumulate_dft_face. */
extern "C" __global__ void accumulate_dft(
    int Nx, int Ny, int Nz,
    int ix0, int ix1,
    int iy0, int iy1,
    int iz0, int iz1,
    real phase_real, real phase_imag,
    const real *field,
    real2 *field_dft
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

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
 *   x-faces: (nzf, nyf) -> abs (ix, iy0+v, iz0+u)
 *   y-faces: (nzf, nxf) -> abs (ix0+v, iy, iz0+u)
 *   z-faces: (nyf, nxf) -> abs (ix0+v, iy0+u, iz)
 */
extern "C" __global__ void accumulate_dft_face(
    int Nx, int Ny, int Nz,
    int face_id,
    int ix0, int ix1,
    int iy0, int iy1,
    int iz0, int iz1,
    int face_offset,
    real phase_real, real phase_imag,
    const real *field,
    real2 *face_dft
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
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

__device__ __forceinline__ void dft_add(real2 *slot, real val, real pr, real pi) {
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
extern "C" __global__ void accumulate_dft_faces_fused(
    int Nx, int Ny, int Nz,
    int ix0, int ix1,
    int iy0, int iy1,
    int iz0, int iz1,
    int off0, int off1, int off2, int off3, int off4, int off5,
    int n_face,
    real phase_e_real, real phase_e_imag,
    real phase_h_real, real phase_h_imag,
    const real * __restrict__ Ex,
    const real * __restrict__ Ey,
    const real * __restrict__ Ez,
    const real * __restrict__ Hx,
    const real * __restrict__ Hy,
    const real * __restrict__ Hz,
    real2 * __restrict__ Ex_dft,
    real2 * __restrict__ Ey_dft,
    real2 * __restrict__ Ez_dft,
    real2 * __restrict__ Hx_dft,
    real2 * __restrict__ Hy_dft,
    real2 * __restrict__ Hz_dft
) {
    int face_i = blockIdx.x * blockDim.x + threadIdx.x;
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

    real pr_e = phase_e_real, pi_e = phase_e_imag;
    real pr_h = phase_h_real, pi_h = phase_h_imag;

/* Co-locate tangential E/H at the face-center (half-cell averages), then DFT.
 * H uses phase_h = phase_e * exp(-j omega dt/2): at monitor time E is at
 * integer t while H is still at (n+1/2)dt after the leapfrog update. */
#define AT(F, ii, jj, kk) ((F)[((ii) * Ny + (jj)) * Nz + (kk)])
#define AVG2(a, b, ok) ((ok) ? (real)0.5 * ((a) + (b)) : (a))
    if (face <= 1) {
        /* Face center (i, j+1/2, k+1/2): avg Ey in z, Ez in y; H from both sides of face. */
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
        /* Face center (i+1/2, j, k+1/2). */
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
        /* Face center (i+1/2, j+1/2, k). */
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
 * Block-local reduction in dynamic shared memory (2 * blockDim.x reals);
 * host sums partials (tiny download).
 */
extern "C" __global__ void dft_rel_change_partial(
    int n,
    const real2 * __restrict__ Ex_c,
    const real2 * __restrict__ Ey_c,
    const real2 * __restrict__ Ez_c,
    const real2 * __restrict__ Hx_c,
    const real2 * __restrict__ Hy_c,
    const real2 * __restrict__ Hz_c,
    const real2 * __restrict__ Ex_p,
    const real2 * __restrict__ Ey_p,
    const real2 * __restrict__ Ez_p,
    const real2 * __restrict__ Hx_p,
    const real2 * __restrict__ Hy_p,
    const real2 * __restrict__ Hz_p,
    real * __restrict__ partial_num,
    real * __restrict__ partial_den
) {
    extern __shared__ unsigned char smem_rel[];
    real *sn = (real *)smem_rel;
    real *sd = sn + blockDim.x;

    int lid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int grp = blockIdx.x;
    real nsum = (real)0;
    real dsum = (real)0;
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
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            sn[lid] += sn[lid + stride];
            sd[lid] += sd[lid + stride];
        }
        __syncthreads();
    }
    if (lid == 0) {
        partial_num[grp] = sn[0];
        partial_den[grp] = sd[0];
    }
}

__device__ __forceinline__ real2 cmul(real2 a, real2 b) {
    return make_real2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ void caccum(real2 *acc, real2 val) {
    acc->x += val.x;
    acc->y += val.y;
}

__device__ __forceinline__ void atomic_add_real2(real2 *addr, real2 val) {
    real *p = (real *)addr;
    atomicAdd(&p[0], val.x);
    atomicAdd(&p[1], val.y);
}

/* Contribution of one packed face sample into N,L (6 real2). */
__device__ void face_sample_NL(
    int face_i,
    real rx, real ry, real rz,
    real k_wave, real dA, real dl,
    int ix0, int ix1, int iy0, int iy1, int iz0, int iz1,
    int off0, int off1, int off2, int off3, int off4, int off5,
    const real2 *Ex_f,
    const real2 *Ey_f,
    const real2 *Ez_f,
    const real2 *Hx_f,
    const real2 *Hy_f,
    const real2 *Hz_f,
    real2 *Nx, real2 *Ny, real2 *Nz,
    real2 *Lx, real2 *Ly, real2 *Lz
) {
    int nxf = ix1 - ix0 + 1;
    int nyf = iy1 - iy0 + 1;
    int nzf = iz1 - iz0 + 1;
    int face, loc, abs_i, abs_j, abs_k;
    real nf, xp, yp, zp;

    if (face_i < off1) {
        face = 0; loc = face_i - off0; nf = (real)-1;
        abs_i = ix0; abs_j = iy0 + loc / nzf; abs_k = iz0 + (loc - (loc / nzf) * nzf);
    } else if (face_i < off2) {
        face = 1; loc = face_i - off1; nf = (real)1;
        abs_i = ix1; abs_j = iy0 + loc / nzf; abs_k = iz0 + (loc - (loc / nzf) * nzf);
    } else if (face_i < off3) {
        face = 2; loc = face_i - off2; nf = (real)-1;
        abs_i = ix0 + loc / nzf; abs_j = iy0; abs_k = iz0 + (loc - (loc / nzf) * nzf);
    } else if (face_i < off4) {
        face = 3; loc = face_i - off3; nf = (real)1;
        abs_i = ix0 + loc / nzf; abs_j = iy1; abs_k = iz0 + (loc - (loc / nzf) * nzf);
    } else if (face_i < off5) {
        face = 4; loc = face_i - off4; nf = (real)-1;
        abs_i = ix0 + loc / nyf; abs_j = iy0 + (loc - (loc / nyf) * nyf); abs_k = iz0;
    } else {
        face = 5; loc = face_i - off5; nf = (real)1;
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
    real2 ph = make_real2(cos(phase), sin(phase));

    /* Trapezoidal face quadrature: half weight on edges, quarter on corners. */
    real wj, wk;
    if (face <= 1) {
        wj = (abs_j == iy0 || abs_j == iy1) ? (real)0.5 : (real)1;
        wk = (abs_k == iz0 || abs_k == iz1) ? (real)0.5 : (real)1;
    } else if (face <= 3) {
        wj = (abs_i == ix0 || abs_i == ix1) ? (real)0.5 : (real)1;
        wk = (abs_k == iz0 || abs_k == iz1) ? (real)0.5 : (real)1;
    } else {
        wj = (abs_i == ix0 || abs_i == ix1) ? (real)0.5 : (real)1;
        wk = (abs_j == iy0 || abs_j == iy1) ? (real)0.5 : (real)1;
    }
    real dAw = dA * wj * wk;

    if (face <= 1) {
        real2 Jy = cmul(make_real2( nf * Hz_f[li].x,  nf * Hz_f[li].y), ph);
        real2 Jz = cmul(make_real2(-nf * Hy_f[li].x, -nf * Hy_f[li].y), ph);
        real2 My = cmul(make_real2(-nf * Ez_f[li].x, -nf * Ez_f[li].y), ph);
        real2 Mz = cmul(make_real2( nf * Ey_f[li].x,  nf * Ey_f[li].y), ph);
        caccum(Ny, make_real2(Jy.x * dAw, Jy.y * dAw));
        caccum(Nz, make_real2(Jz.x * dAw, Jz.y * dAw));
        caccum(Ly, make_real2(My.x * dAw, My.y * dAw));
        caccum(Lz, make_real2(Mz.x * dAw, Mz.y * dAw));
    } else if (face <= 3) {
        real2 Jx = cmul(make_real2(-nf * Hz_f[li].x, -nf * Hz_f[li].y), ph);
        real2 Jz = cmul(make_real2( nf * Hx_f[li].x,  nf * Hx_f[li].y), ph);
        real2 Mx = cmul(make_real2( nf * Ez_f[li].x,  nf * Ez_f[li].y), ph);
        real2 Mz = cmul(make_real2(-nf * Ex_f[li].x, -nf * Ex_f[li].y), ph);
        caccum(Nx, make_real2(Jx.x * dAw, Jx.y * dAw));
        caccum(Nz, make_real2(Jz.x * dAw, Jz.y * dAw));
        caccum(Lx, make_real2(Mx.x * dAw, Mx.y * dAw));
        caccum(Lz, make_real2(Mz.x * dAw, Mz.y * dAw));
    } else {
        real2 Jx = cmul(make_real2( nf * Hy_f[li].x,  nf * Hy_f[li].y), ph);
        real2 Jy = cmul(make_real2(-nf * Hx_f[li].x, -nf * Hx_f[li].y), ph);
        real2 Mx = cmul(make_real2(-nf * Ey_f[li].x, -nf * Ey_f[li].y), ph);
        real2 My = cmul(make_real2( nf * Ex_f[li].x,  nf * Ex_f[li].y), ph);
        caccum(Nx, make_real2(Jx.x * dAw, Jx.y * dAw));
        caccum(Ny, make_real2(Jy.x * dAw, Jy.y * dAw));
        caccum(Lx, make_real2(Mx.x * dAw, Mx.y * dAw));
        caccum(Ly, make_real2(My.x * dAw, My.y * dAw));
    }
}

/*
 * Parallel over face samples (dim0) and observation points (dim1).
 * Local reduction along dim0 in dynamic shared memory (6 * blockDim.x real2),
 * then atomic add into NL[obs*6 + c].
 */
extern "C" __global__ void farfield_accumulate_nl(
    int n_face,
    int n_obs,
    const real *obs_xyz,
    real k_wave,
    real dl,
    int ix0, int ix1,
    int iy0, int iy1,
    int iz0, int iz1,
    int off0, int off1, int off2, int off3, int off4, int off5,
    const real2 *Ex_f,
    const real2 *Ey_f,
    const real2 *Ez_f,
    const real2 *Hx_f,
    const real2 *Hy_f,
    const real2 *Hz_f,
    real2 *NL_out
) {
    extern __shared__ unsigned char smem_nl[];
    real2 *scratch = (real2 *)smem_nl;

    int face_i = blockIdx.x * blockDim.x + threadIdx.x;
    int obs = blockIdx.y * blockDim.y + threadIdx.y;
    int lid = threadIdx.x;
    int lsize = blockDim.x;

    real2 Nx = make_real2((real)0, (real)0);
    real2 Ny = make_real2((real)0, (real)0);
    real2 Nz = make_real2((real)0, (real)0);
    real2 Lx = make_real2((real)0, (real)0);
    real2 Ly = make_real2((real)0, (real)0);
    real2 Lz = make_real2((real)0, (real)0);

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
    __syncthreads();

    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            for (int c = 0; c < 6; c++) {
                int a = c * lsize + lid;
                scratch[a].x += scratch[a + stride].x;
                scratch[a].y += scratch[a + stride].y;
            }
        }
        __syncthreads();
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

/* Convert integrated N,L -> far E,H for each observation. */
extern "C" __global__ void farfield_nl_to_eh(
    int n_obs,
    const real *obs_xyz,
    real k_wave,
    real eta0,
    const real2 *NL_in,
    real2 *EH_out
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
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
    real2 eikr = make_real2(cos(ang), -sin(ang));
    real scale = k_wave / ((real)4.0 * (real)3.14159265358979323846 * r);
    real2 pref = cmul(make_real2((real)0, -scale), eikr);

    real2 rxL[3], Nt[3], E[3], H[3];
    rxL[0] = make_real2(rhat[1] * Lvec[2].x - rhat[2] * Lvec[1].x,
                        rhat[1] * Lvec[2].y - rhat[2] * Lvec[1].y);
    rxL[1] = make_real2(rhat[2] * Lvec[0].x - rhat[0] * Lvec[2].x,
                        rhat[2] * Lvec[0].y - rhat[0] * Lvec[2].y);
    rxL[2] = make_real2(rhat[0] * Lvec[1].x - rhat[1] * Lvec[0].x,
                        rhat[0] * Lvec[1].y - rhat[1] * Lvec[0].y);

    real2 Ndot = make_real2(
        rhat[0] * Nvec[0].x + rhat[1] * Nvec[1].x + rhat[2] * Nvec[2].x,
        rhat[0] * Nvec[0].y + rhat[1] * Nvec[1].y + rhat[2] * Nvec[2].y);

    for (int c = 0; c < 3; c++) {
        Nt[c] = make_real2(Nvec[c].x - Ndot.x * rhat[c], Nvec[c].y - Ndot.y * rhat[c]);
        real2 tE = make_real2(eta0 * Nt[c].x + rxL[c].x, eta0 * Nt[c].y + rxL[c].y);
        E[c] = cmul(pref, tE);
    }
    // Far-field TEM: H = -r_hat x E / eta
    H[0] = make_real2(-(rhat[1] * E[2].x - rhat[2] * E[1].x) / eta0,
                      -(rhat[1] * E[2].y - rhat[2] * E[1].y) / eta0);
    H[1] = make_real2(-(rhat[2] * E[0].x - rhat[0] * E[2].x) / eta0,
                      -(rhat[2] * E[0].y - rhat[0] * E[2].y) / eta0);
    H[2] = make_real2(-(rhat[0] * E[1].x - rhat[1] * E[0].x) / eta0,
                      -(rhat[0] * E[1].y - rhat[1] * E[0].y) / eta0);

    int o = 6 * p;
    EH_out[o + 0] = E[0];
    EH_out[o + 1] = E[1];
    EH_out[o + 2] = E[2];
    EH_out[o + 3] = H[0];
    EH_out[o + 4] = H[1];
    EH_out[o + 5] = H[2];
}
