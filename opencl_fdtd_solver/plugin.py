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
    _pending_currents = None

    def _inject_current(self, injection):
        if self._pending_currents is None:
            injection()
        else:
            self._pending_currents.append(injection)

    def _step_fields(self):
        """Keep callback timing while adding J after decay of the old E field."""
        self._update_H()
        t_int = self.t
        self._pending_currents = []
        try:
            self.t = t_int + 0.5 * self.dt
            for src in self._sources:
                src(self)
        finally:
            self.t = t_int
            currents = self._pending_currents
            self._pending_currents = None
        self._update_E()
        for injection in currents:
            injection()

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
