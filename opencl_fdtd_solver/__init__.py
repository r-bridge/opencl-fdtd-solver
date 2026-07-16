__version__ = "1.0.0"
__name__ = "opencl_fdtd_solver"

from .engine import OpenCLFDTD
from .monitors import NumPyNear2FarMonitor, OpenCLNear2FarMonitor
from .numpy_engine import NumPyFDTD
from .plugin import StepCallback

__all__ = [
    "OpenCLFDTD",
    "NumPyFDTD",
    "NumPyNear2FarMonitor",
    "OpenCLNear2FarMonitor",
    "StepCallback",
]
