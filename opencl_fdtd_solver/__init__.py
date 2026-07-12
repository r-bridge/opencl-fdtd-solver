__version__ = "1.0.0"
__name__ = "opencl_fdtd_solver"

from .engine import OpenCLFDTD
from .numpy_engine import NumPyFDTD
from .monitors import NumPyNear2FarMonitor, OpenCLNear2FarMonitor
