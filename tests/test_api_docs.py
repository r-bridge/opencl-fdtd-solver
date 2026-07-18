# Copyright (C) 2026: OpenCL FDTD Solver Contributors
#
# This file is part of opencl-fdtd-solver.

"""Guard: docs/API.md must mention every public export and public method."""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import opencl_fdtd_solver as pkg
from opencl_fdtd_solver import (
    NumPyFDTD,
    NumPyFDTD_FaceCPML,
    NumPyNear2FarMonitor,
    OpenCLFDTD,
    OpenCLNear2FarMonitor,
    StepCallback,
)

_API_MD = Path(__file__).resolve().parents[1] / "docs" / "API.md"

# Classes whose public (non-underscore) callables must appear in API.md.
_DOCUMENTED_TYPES = (
    OpenCLFDTD,
    NumPyFDTD,
    NumPyFDTD_FaceCPML,
    OpenCLNear2FarMonitor,
    NumPyNear2FarMonitor,
    StepCallback,
)


def _public_callables(cls) -> list[str]:
    names: list[str] = []
    for name, obj in inspect.getmembers(cls):
        if name.startswith("_"):
            continue
        if name == "mro":
            continue
        if inspect.isroutine(obj) or isinstance(obj, (staticmethod, classmethod, property)):
            names.append(name)
        elif callable(obj):
            names.append(name)
    return sorted(set(names))


class TestApiDocs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = _API_MD.read_text(encoding="utf-8")
        cls.assertTrue(_API_MD.is_file(), f"missing {_API_MD}")

    def test_all_exports_mentioned(self):
        for name in pkg.__all__:
            if name == "__version__":
                continue
            self.assertIn(
                name,
                self.text,
                f"{name!r} is in opencl_fdtd_solver.__all__ but missing from docs/API.md",
            )

    def test_public_methods_mentioned(self):
        for cls in _DOCUMENTED_TYPES:
            for name in _public_callables(cls):
                self.assertIn(
                    name,
                    self.text,
                    f"{cls.__name__}.{name} is public but missing from docs/API.md",
                )

    def test_agents_md_points_at_api_docs(self):
        agents = Path(__file__).resolve().parents[1] / "AGENTS.md"
        self.assertTrue(agents.is_file(), "AGENTS.md missing")
        body = agents.read_text(encoding="utf-8")
        self.assertIn("docs/API.md", body)
        self.assertIn("test_api_docs", body)


if __name__ == "__main__":
    unittest.main()
