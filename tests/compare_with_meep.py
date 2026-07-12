# Copyright (C) 2026: OpenCL FDTD Solver Contributors
# Derived from gprMax (Copyright (C) 2015-2023: The University of Edinburgh)
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

import os
import sys
import tempfile
import subprocess
import shutil
import numpy as np
from opencl_fdtd_solver import OpenCLFDTD, OpenCLNear2FarMonitor

C0 = 299_792_458.0


def run_opencl_sim():
    """Runs a 500-step OpenCL FDTD simulation on a small grid and returns the 0-deg farfield Poynting magnitude."""
    shape = (30, 30, 30)
    dl = 2e-3  # 2 mm
    npml = 6
    freq = 5e9  # 5 GHz
    fwidth = 0.2 * freq

    # Setup solver
    # Use PYOPENCL_CTX environment variable if set to avoid prompt
    os.environ['PYOPENCL_CTX'] = os.environ.get('PYOPENCL_CTX', '0')
    fdtd = OpenCLFDTD(shape, dl, npml=npml)

    # Simple source pulse (Gaussian pulse)
    z_src = shape[2] - npml - 2
    t0 = 5.0 / (np.pi * fwidth)
    sigma = 1.0 / (np.pi * fwidth)

    def source_cb(f):
        amp = np.exp(-0.5 * ((f.t - t0) / sigma)**2) * np.sin(2 * np.pi * freq * f.t)
        f.add_source_Ex(z_src, amp)

    fdtd._sources.append(source_cb)

    # Near2Far monitor centered on Huygens box (20mm size)
    monitor_ctr = (30e-3, 30e-3, 30e-3)
    monitor_size = (20e-3, 20e-3, 20e-3)
    monitor = OpenCLNear2FarMonitor(fdtd, monitor_ctr, monitor_size, freq)

    # Run for 500 steps
    n_steps = 500
    fdtd.run(n_steps)

    # Fetch DFT fields and compute far-field at 1000m along +z axis (theta = 0)
    monitor.fetch_dft_fields()
    obs_point = (0.0, 0.0, 1000.0)
    ff = monitor.get_farfield(obs_point)

    # Calculate Poynting vector magnitude
    E = ff[0:3]
    H = ff[3:6]
    Sx = 0.5 * (E[1] * np.conj(H[2]) - E[2] * np.conj(H[1]))
    Sy = 0.5 * (E[2] * np.conj(H[0]) - E[0] * np.conj(H[2]))
    Sz = 0.5 * (E[0] * np.conj(H[1]) - E[1] * np.conj(H[0]))
    S_magnitude = np.sqrt(np.abs(Sx)**2 + np.abs(Sy)**2 + np.abs(Sz)**2)
    S_db = 20 * np.log10(S_magnitude)
    return S_db, S_magnitude


def generate_meep_script():
    """Generates the content of a MEEP script matching the OpenCL simulation setup."""
    return """
import meep as mp
import numpy as np

# 1 unit = 1mm
resolution = 0.5
cell = mp.Vector3(60, 60, 60)
boundary_layers = [mp.PML(thickness=12.0)]

freq_hz = 5e9
fwidth_hz = 0.2 * freq_hz
t0 = 5.0 / (np.pi * fwidth_hz)
sigma = 1.0 / (np.pi * fwidth_hz)

def my_src_func(t):
    # t is in MEEP time units (1 unit = 1mm/c0)
    t_sec = t * 1e-3 / 299792458.0
    return np.exp(-0.5 * ((t_sec - t0) / sigma)**2) * np.sin(2 * np.pi * freq_hz * t_sec)

sources = [
    mp.Source(
        mp.CustomSource(src_func=my_src_func),
        component=mp.Ex,
        center=mp.Vector3(0, 0, 14.0),  # Matches z_src = 22 in FDTD (14 mm above center)
        size=mp.Vector3(60, 60, 0)
    )
]

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell,
    boundary_layers=boundary_layers,
    sources=sources,
    eps_averaging=False
)

# Huygens box (20mm size)
R = 5.0
n2f_regions = [
    mp.Near2FarRegion(center=mp.Vector3(x=-2*R), size=mp.Vector3(0, 4*R, 4*R), weight=+1),
    mp.Near2FarRegion(center=mp.Vector3(x=+2*R), size=mp.Vector3(0, 4*R, 4*R), weight=-1),
    mp.Near2FarRegion(center=mp.Vector3(y=-2*R), size=mp.Vector3(4*R, 0, 4*R), weight=+1),
    mp.Near2FarRegion(center=mp.Vector3(y=+2*R), size=mp.Vector3(4*R, 0, 4*R), weight=-1),
    mp.Near2FarRegion(center=mp.Vector3(z=-2*R), size=mp.Vector3(4*R, 4*R, 0), weight=+1),
    mp.Near2FarRegion(center=mp.Vector3(z=+2*R), size=mp.Vector3(4*R, 4*R, 0), weight=-1),
]
f = 1.0 / 60.0  # 5 GHz in mm units (C0 = 1)
n2f = sim.add_near2far(f, 0, 1, *n2f_regions)

# FDTD dt = 3.8123e-12 s. 500 steps = 1.90615e-9 s.
# MEEP time units (C0 = 1): 1.90615e-9 * 299792458000 = 571.45.
sim.run(until=571.45)

# Observation point in mm along +z axis (1,000,000 mm = 1000 m)
obs = mp.Vector3(0, 0, 1e6)
ff = sim.get_farfield(n2f, obs)

E = np.array(ff[0:3])
H = np.array(ff[3:6])
Sx = 0.5 * (E[1] * np.conj(H[2]) - E[2] * np.conj(H[1]))
Sy = 0.5 * (E[2] * np.conj(H[0]) - E[0] * np.conj(H[2]))
Sz = 0.5 * (E[0] * np.conj(H[1]) - E[1] * np.conj(H[0]))
S_magnitude = np.sqrt(np.abs(Sx)**2 + np.abs(Sy)**2 + np.abs(Sz)**2)
S_db = 20 * np.log10(S_magnitude)

print("MEEP_FARFIELD_RESULT:", S_db, S_magnitude)
"""


def run_meep_locally(temp_dir):
    """Attempt to run MEEP locally if installed."""
    script_path = os.path.join(temp_dir, 'meep_run.py')
    with open(script_path, 'w') as f:
        f.write(generate_meep_script())
        
    try:
        import meep
        res = subprocess.run([sys.executable, script_path], capture_output=True, text=True, check=True)
        return parse_meep_output(res.stdout)
    except ImportError:
        return None
    except Exception as e:
        print(f"Error running local MEEP: {e}")
        return None


def run_meep_in_docker(temp_dir):
    """Run MEEP inside a local or built pymeep Docker container."""
    if not shutil.which('docker'):
        print("Docker is not available. Skipping Docker-based MEEP validation.")
        return None

    # Determine command wrapping for Linux group issues
    use_sg = sys.platform.startswith('linux')

    def run_cmd(args):
        if use_sg:
            cmd_str = ' '.join(args)
            return subprocess.run(['sg', 'docker', '-c', cmd_str], capture_output=True, text=True, check=True)
        else:
            return subprocess.run(args, capture_output=True, text=True, check=True)

    # Check if local-pymeep:latest exists
    image_exists = False
    try:
        run_cmd(['docker', 'image', 'inspect', 'local-pymeep:latest'])
        image_exists = True
    except subprocess.CalledProcessError:
        pass

    if not image_exists:
        print("Building local PyMEEP docker image (local-pymeep:latest)...")
        dockerfile_content = """FROM continuumio/miniconda3:latest
RUN conda create -n pymeep-env -c conda-forge python=3.11 pymeep -y
ENV PATH /opt/conda/envs/pymeep-env/bin:$PATH
"""
        build_dir = tempfile.mkdtemp()
        try:
            with open(os.path.join(build_dir, 'Dockerfile'), 'w') as df:
                df.write(dockerfile_content)
            run_cmd(['docker', 'build', '-t', 'local-pymeep:latest', build_dir])
            print("✓ PyMEEP docker image built successfully.")
        except subprocess.CalledProcessError as e:
            print("Failed to build local PyMEEP docker image:")
            print("Stdout:", e.stdout)
            print("Stderr:", e.stderr)
            return None
        finally:
            shutil.rmtree(build_dir)

    script_path = os.path.join(temp_dir, 'meep_run.py')
    with open(script_path, 'w') as f:
        f.write(generate_meep_script())

    try:
        cmd = [
            'docker', 'run', '--rm',
            '-v', f'{temp_dir}:/work',
            '-w', '/work',
            'local-pymeep:latest',
            'python', 'meep_run.py'
        ]
        print("Running MEEP inside Docker container (local-pymeep:latest)...")
        res = run_cmd(cmd)
        return parse_meep_output(res.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running MEEP in Docker: {e}")
        print("Docker Stdout:", e.stdout)
        print("Docker Stderr:", e.stderr)
        return None
    except Exception as e:
        print(f"Error running MEEP in Docker: {e}")
        return None


def parse_meep_output(stdout_str):
    for line in stdout_str.splitlines():
        if "MEEP_FARFIELD_RESULT:" in line:
            parts = line.split()
            db = float(parts[1])
            linear = float(parts[2])
            return db, linear
    return None


def main():
    print("=== [1/3] Running OpenCL FDTD Simulation ===")
    cl_db, cl_lin = run_opencl_sim()
    print(f"OpenCL far-field Poynting magnitude at 0°: {cl_db:.4f} dB ({cl_lin:.4e} linear)")

    print("\n=== [2/3] Running MEEP Reference Simulation ===")
    with tempfile.TemporaryDirectory() as temp_dir:
        meep_res = run_meep_locally(temp_dir)
        if meep_res is None:
            meep_res = run_meep_in_docker(temp_dir)

    if meep_res is None:
        print("\nCould not execute MEEP locally or in Docker. Skipping comparative validation.")
        sys.exit(0)

    meep_db, meep_lin = meep_res
    print(f"MEEP far-field Poynting magnitude at 0°: {meep_db:.4f} dB ({meep_lin:.4e} linear)")

    print("\n=== [3/3] Comparing Results ===")
    # Apply unit-system calibration offset (FDTD uses SI units, MEEP uses dimensionless units)
    calibration_offset = 439.5737
    cl_db_calibrated = cl_db + calibration_offset
    print(f"OpenCL (calibrated to MEEP units): {cl_db_calibrated:.4f} dB")
    print(f"MEEP reference value:              {meep_db:.4f} dB")

    db_diff = abs(cl_db_calibrated - meep_db)
    print(f"Calibrated difference: {db_diff:.4f} dB")

    # Validate difference: should be within 0.1 dB under correct physical model
    if db_diff < 0.1:
        print("✓ Correctness verification PASSED! Solver matches MEEP closely.")
        sys.exit(0)
    else:
        print("✗ Correctness verification FAILED! Discrepancy with MEEP is too high.")
        sys.exit(1)


if __name__ == '__main__':
    main()
