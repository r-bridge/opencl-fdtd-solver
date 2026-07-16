__version__ = "1.0.0"
__name__ = "opencl_fdtd_solver"

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
    "NumPyFDTD",
    "NumPyNear2FarMonitor",
    "OpenCLNear2FarMonitor",
    "StepCallback",
]
