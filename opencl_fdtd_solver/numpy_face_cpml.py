# Copyright (C) 2026: OpenCL FDTD Solver Contributors
#
# This file is part of opencl-fdtd-solver.
#
# opencl-fdtd-solver is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# opencl-fdtd-solver is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with opencl-fdtd-solver.  If not, see <http://www.gnu.org/licenses/>.

"""
Experimental NumPy FDTD variant with face-local CPML auxiliary fields.

Goal: match the OpenCL engine's memory layout (psi only at boundary faces)
so large CPU reference runs don't require full-volume auxiliary arrays.

Status: scaffolding class that currently delegates to the base implementation.
This preserves test behavior while we implement the face-local buffers and
striped updates incrementally.
"""

from __future__ import annotations

from .numpy_engine import NumPyFDTD


class NumPyFDTD_FaceCPML(NumPyFDTD):
    """
    NumPy FDTD with planned face-local CPML storage.

    For now, this class inherits the volume-CPML behavior from NumPyFDTD
    to keep tests green. Subsequent commits will:
      - allocate psi buffers only on x/y/z PML face stripes
      - update H/E using striped psi contributions
      - keep interior updates fast with the existing helpers
    """

    # Placeholder to allow importers to opt-in to the new layout later:
    CPML_STORAGE = "face"

    # The step(), run(), and API remain identical; updates are delegated.
    # Implementation will be filled in in a follow-up change.

