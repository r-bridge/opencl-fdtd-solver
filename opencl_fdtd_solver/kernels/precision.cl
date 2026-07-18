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
