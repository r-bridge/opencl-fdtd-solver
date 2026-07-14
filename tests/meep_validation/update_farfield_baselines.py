# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
#
# This file is part of opencl-fdtd-solver.

"""Regenerate near-to-far golden patterns, EH arrays, and discrepancy reports.

Usage (repo root)::

    IGNORE_GPU=NVIDIA,AMD PYOPENCL_CTX=0 python -m tests.meep_validation.update_farfield_baselines

Re-writes under ``tests/meep_validation/baselines/``:

- per-case ``pattern_xz.png`` / ``*.npy`` / ``meta.json``
- aggregate ``DISCREPANCY_REPORT_FARFIELD.md`` and ``discrepancy_report_farfield.json``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .farfield_cases import default_farfield_cases, write_all_farfield_baselines
from .plane_cases import baselines_root


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Optional case names (default: all).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Baseline root (default: tests/meep_validation/baselines).",
    )
    args = p.parse_args(argv)
    root = baselines_root() if args.out is None else Path(args.out)
    cases = default_farfield_cases()
    if args.cases:
        want = set(args.cases)
        cases = [c for c in cases if c.name in want]
        missing = want - {c.name for c in cases}
        if missing:
            print(f"Unknown cases: {sorted(missing)}", file=sys.stderr)
            return 2
    print(f"Updating {len(cases)} far-field case(s) under {root} …", flush=True)
    write_all_farfield_baselines(root, cases)
    print(f"  wrote {root / 'DISCREPANCY_REPORT_FARFIELD.md'}", flush=True)
    print(f"  wrote {root / 'discrepancy_report_farfield.json'}", flush=True)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
