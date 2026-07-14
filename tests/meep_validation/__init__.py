# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
#
# This file is part of opencl-fdtd-solver.

"""Re-export harness helpers for `tests.meep_validation`."""

from .harness import (  # noqa: F401
    C0,
    OPENCL_COURANT,
    MeepUnavailableError,
    complex_align,
    ensure_pyopencl_ctx,
    gaussian_sine_amp,
    max_abs_db_error,
    meep_until,
    opencl_dt,
    parse_meep_json,
    peak_normalize,
    poynting_db_from_eh,
    rel_mag_error,
    run_meep_script,
)
