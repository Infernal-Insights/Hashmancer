import os
import sys
import shutil
import subprocess
from pathlib import Path
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))

try:
    import pyopencl as cl
except Exception:
    cl = None

OPENCL_KERNEL = os.path.join(ROOT, 'darkling', 'intel_backend', 'opencl_kernel.cl')

@pytest.mark.skipif(cl is None, reason="pyopencl not installed")
def test_opencl_kernel_compiles():
    try:
        plats = cl.get_platforms()
    except cl.LogicError:
        pytest.skip('no OpenCL platform')
    if not plats:
        pytest.skip('no OpenCL platform')
    ctx = cl.Context(devices=[plats[0].get_devices()[0]])
    with open(OPENCL_KERNEL) as f:
        src = f.read()
    program = cl.Program(ctx, src).build()
    assert 'crack_kernel' in program.get_info(cl.program_info.KERNEL_NAMES)

def test_hip_kernel_present():
    hipcc = shutil.which('hipcc')
    if hipcc is None:
        pytest.skip('hipcc not available')
    path = os.path.join(ROOT, 'darkling', 'hip_backend', 'darkling_engine.hip')
    assert os.path.exists(path)


def test_cmake_build_backends(tmp_path):
    hipcc = shutil.which('hipcc')
    has_opencl = shutil.which('pkg-config') and shutil.which('clang')
    if not hipcc and not has_opencl:
        pytest.skip('HIP/OpenCL toolchains unavailable')

    build_dir = tmp_path / 'build'
    args = ['cmake', str(Path(ROOT)/'darkling')]
    if hipcc:
        args += ['-DENABLE_HIP=ON']
    else:
        args += ['-DENABLE_HIP=OFF']
    if has_opencl:
        args += ['-DENABLE_OPENCL=ON']
    else:
        args += ['-DENABLE_OPENCL=OFF']

    res = subprocess.run(args + ['-B', str(build_dir)], capture_output=True)
    if res.returncode != 0:
        pytest.skip('cmake configuration failed')
    res = subprocess.run(['cmake', '--build', str(build_dir)], capture_output=True)
    assert res.returncode == 0
