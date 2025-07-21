import os
import sys
import shutil
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

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
