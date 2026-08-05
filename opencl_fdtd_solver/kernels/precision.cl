/* Copyright (C) 2026: OpenCL FDTD Solver Contributors
 *
 * Precision typedefs for FP32/FP64 dual builds.
 * Host builds with -DUSE_FP64=1 for double; default is float.
 */

#ifndef USE_FP64
#define USE_FP64 0
#endif

#if USE_FP64
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "OpenCL FP64 requested but cl_khr_fp64 / cl_amd_fp64 is unavailable"
#endif
#if defined(cl_khr_int64_base_atomics)
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif
typedef double real;
typedef double2 real2;
#else
typedef float real;
typedef float2 real2;
#endif

/*
 * accreal/accreal2: independent precision for *accumulator* state (DFT face
 * buffers, near-to-far N/L reduction) — decoupled from the volumetric field
 * precision above. Default: always double, regardless of USE_FP64, since a
 * 12k-timestep running DFT sum and a ~1e6-sample atomic N/L reduction are
 * exactly the kind of repeated-summation ops that compound fp32 round-off,
 * while costing negligible extra memory (accumulators are sized by monitor
 * face-sample count, not the full 3D grid). Set ACCUM_FP64=0 to fall back
 * to `real`/`real2` (matches old single-precision-everywhere behavior).
 */
#ifndef ACCUM_FP64
#define ACCUM_FP64 1
#endif

#if ACCUM_FP64
#if !USE_FP64
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "ACCUM_FP64 requested but cl_khr_fp64 / cl_amd_fp64 is unavailable"
#endif
#if defined(cl_khr_int64_base_atomics)
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif
#endif
typedef double accreal;
typedef double2 accreal2;
#else
typedef real accreal;
typedef real2 accreal2;
#endif
