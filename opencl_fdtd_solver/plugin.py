# Copyright (C) 2026: OpenCL FDTD Solver Contributors
#
# This file is part of opencl-fdtd-solver.

"""Public source/monitor registration for FDTD solvers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StepCallback(Protocol):
    """Callable invoked once per timestep with the solver instance.

    Sources run after the H update and before the E update.
    Monitors run after the full Yee step (fields and ``t`` advanced).
    """

    def __call__(self, fdtd: Any) -> None: ...


class SourceMonitorMixin:
    """Public registration API for timestep sources and monitors.

    Concrete solvers must initialize ``_sources`` and ``_monitors`` lists in
    ``__init__``. Prefer these helpers over touching the private lists.
    """

    _sources: list
    _monitors: list

    def add_source(self, source: StepCallback) -> StepCallback:
        """Register a source callback ``source(fdtd)`` after each H update."""
        if not callable(source):
            raise TypeError(f"source must be callable, got {type(source)!r}")
        self._sources.append(source)
        return source

    def add_monitor(self, monitor: StepCallback) -> StepCallback:
        """Register a monitor callback ``monitor(fdtd)`` after each full step."""
        if not callable(monitor):
            raise TypeError(f"monitor must be callable, got {type(monitor)!r}")
        self._monitors.append(monitor)
        return monitor

    def clear_sources(self) -> None:
        """Remove all registered source callbacks."""
        self._sources.clear()

    def clear_monitors(self) -> None:
        """Remove all registered monitor callbacks."""
        self._monitors.clear()
