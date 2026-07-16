__version__ = "1.0.0"

from .constants import C0, EPS0, ETA0, MU0
from .engine import OpenCLFDTD
from .monitors import NumPyNear2FarMonitor, OpenCLNear2FarMonitor
from .numpy_engine import NumPyFDTD
from .plugin import StepCallback

__all__ = [
    "C0",
    "MU0",
    "EPS0",
    "ETA0",
    "OpenCLFDTD",
    "CUDAFDTD",
    "NumPyFDTD",
    "NumPyNear2FarMonitor",
    "OpenCLNear2FarMonitor",
    "CUDANear2FarMonitor",
    "StepCallback",
    "__version__",
]

# CUDA engine is exported lazily: importing it requires the optional cupy
# dependency, which must not be a hard requirement of the base package.
_CUDA_EXPORTS = {
    "CUDAFDTD": ("opencl_fdtd_solver.cuda_engine", "CUDAFDTD"),
    "CUDANear2FarMonitor": ("opencl_fdtd_solver.cuda_monitors", "CUDANear2FarMonitor"),
}


def __getattr__(name):
    target = _CUDA_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value
