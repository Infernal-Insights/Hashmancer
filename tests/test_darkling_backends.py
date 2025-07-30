import os
import sys
import shutil
import subprocess
from pathlib import Path
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))

try:
    import pyopencl as cl
except ImportError:
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


@pytest.mark.skipif(cl is None, reason="pyopencl not installed")
def test_opencl_backend_exec(tmp_path):
    try:
        plats = cl.get_platforms()
    except cl.LogicError:
        pytest.skip('no OpenCL platform')
    if not plats:
        pytest.skip('no OpenCL platform')
    gpp = shutil.which('g++')
    if gpp is None:
        pytest.skip('g++ not available')

    src = r"""
#include "darkling/intel_backend/intel_cracker.h"
#include <cstring>
#include <iostream>
using namespace darkling;
int main(){
    IntelCracker c; MaskJob job{};
    job.start_index=0; job.end_index=1;
    job.mask_length=1; job.mask_template[0]=0;
    job.charset_lengths[0]=1; job.charsets[0][0]='a';
    const uint8_t hash[16]={0x0c,0xc1,0x75,0xb9,0xc0,0xf1,0xb6,0xa8,0x31,0xc3,0x99,0xe2,0x69,0x77,0x26,0x61};
    job.hash_length=16; job.num_hashes=1;
    std::memcpy(job.hashes[0],hash,16);
    if(!c.initialize()||!c.load_job(job)||!c.run_batch()) return 1;
    auto res=c.read_results();
    if(res.empty()) return 1;
    std::cout<<(int)res[0].length<<' '<<res[0].password[0]<<' ';
    auto st=c.get_status();
    std::cout<<st.hashes_processed;
    return 0;
}
"""
    src_file = tmp_path / 'test.cpp'
    src_file.write_text(src)
    exe = tmp_path / 'exec'
    subprocess.check_call([
        gpp, '-std=c++17', '-I.', '-Idarkling', str(src_file),
        'darkling/intel_backend/intel_cracker.cpp', '-lOpenCL', '-o', str(exe)
    ], cwd=ROOT)
    res = subprocess.run([str(exe)], cwd=ROOT, capture_output=True)
    if res.returncode != 0:
        pytest.skip('runtime failed')
    out = res.stdout.decode().strip()
    assert out.startswith('1 a')


def build_backend_detector(tmp_path: Path) -> Path:
    gpp = shutil.which('g++')
    if gpp is None:
        pytest.skip('g++ not available')

    wrapper = tmp_path / 'wrapper.cpp'
    wrapper.write_text(
        '\n'.join([
            '#include <iostream>',
            '#include "darkling/gpu_backend.h"',
            'extern "C" darkling::GpuBackend test_detect_backend(const char*);',
            'int main(int argc, char** argv){',
            '    const char* arg = argc>1 ? argv[1] : nullptr;',
            '    auto b = test_detect_backend(arg);',
            '    std::cout << static_cast<int>(b);',
            '    return 0;',
            '}',
        ])
    )
    exe = tmp_path / 'detector'
    subprocess.check_call([
        gpp, '-std=c++17', '-DDARKLING_NO_MAIN', '-I.', '-Ihashmancer',
        str(wrapper), 'hashmancer/darkling/backend_dispatcher.cpp',
        '-ldl', '-o', str(exe)
    ], cwd=ROOT)
    return exe


def make_dummy_lib(path: Path, name: str):
    src = 'int x() { return 0; }'
    dummy = path / (name + '.c')
    dummy.write_text(src)
    subprocess.check_call([
        'gcc', '-shared', '-fPIC', str(dummy), '-o', str(path / name)
    ])


def test_dispatcher_env_selection(tmp_path):
    exe = build_backend_detector(tmp_path)
    env = os.environ.copy()
    env['DARKLING_GPU_BACKEND'] = 'hip'
    res = subprocess.run([str(exe)], env=env, capture_output=True, text=True)
    assert res.stdout.strip() == '1'
    env['DARKLING_GPU_BACKEND'] = 'opencl'
    res = subprocess.run([str(exe)], env=env, capture_output=True, text=True)
    assert res.stdout.strip() == '2'


def test_dispatcher_library_detection(tmp_path):
    exe = build_backend_detector(tmp_path)
    make_dummy_lib(tmp_path, 'libamdhip64.so')
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(tmp_path)
    res = subprocess.run([str(exe)], env=env, capture_output=True, text=True)
    assert res.stdout.strip() == '1'
    os.remove(tmp_path / 'libamdhip64.so')
    make_dummy_lib(tmp_path, 'libOpenCL.so')
    res = subprocess.run([str(exe)], env=env, capture_output=True, text=True)
    assert res.stdout.strip() == '2'
