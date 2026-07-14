# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
#
# This file is part of opencl-fdtd-solver.

"""Run unit tests under coverage, then exit without OpenCL/POCL teardown.

POCL on GitHub Actions often aborts during interpreter shutdown (SIGSEGV or
LLVM errors) after tests have already passed. Saving coverage and calling
``os._exit`` skips those destructors while still enforcing fail-under.
"""

from __future__ import annotations

import os
import sys
import unittest

import coverage


SUITES = [
    "tests.test_solver",
    "tests.test_unit_engine",
    "tests.test_unit_monitors",
    "tests.test_unit_harness",
    "tests.test_unit_plane_render",
    "tests.test_unit_plane_metrics",
    "tests.test_unit_farfield_baselines",
]


def main() -> None:
    # Allow `python tests/run_coverage.py` from the repo root.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    cov = coverage.Coverage(source=["opencl_fdtd_solver"], branch=True)
    cov.start()
    suite = unittest.defaultTestLoader.loadTestsFromNames(SUITES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    cov.stop()
    cov.save()

    ok_tests = result.wasSuccessful()
    total = cov.report(show_missing=True)
    cov.xml_report(outfile="coverage.xml")
    ok_cov = float(total) >= 90.0
    if not ok_cov:
        print(f"ERROR: coverage {total:.1f}% is below the 90% gate", file=sys.stderr)

    # Hard-exit so POCL/LLVM does not segfault during context destruction.
    os._exit(0 if (ok_tests and ok_cov) else 1)


if __name__ == "__main__":
    main()
