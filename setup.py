from setuptools import find_packages, setup

setup(
    name="opencl_fdtd_solver",
    version="1.0.0",
    author="OpenCL FDTD Solver Contributors",
    description="A generic 3D Yee-grid FDTD electromagnetic solver accelerated with OpenCL.",
    license="GPLv3",
    packages=find_packages(),
    package_data={
        "opencl_fdtd_solver": ["kernels/*.cl"],
    },
    include_package_data=True,
    install_requires=[
        "numpy",
        "pyopencl",
    ],
    extras_require={
        "test": [
            "matplotlib",
            "pillow",
            "coverage",
        ],
        "lint": [
            "ruff>=0.8",
            "mypy>=1.8",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.10",
)
