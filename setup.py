from setuptools import setup, find_packages

setup(
    name="opencl_fdtd_solver",
    version="1.0.0",
    author="OpenCL FDTD Solver Contributors",
    description="A generic 3D Yee-grid FDTD electromagnetic solver accelerated with OpenCL.",
    license="GPLv3",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pyopencl",
        "h5py",
        "scipy",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering"
    ],
    python_requires=">=3.6",
)
