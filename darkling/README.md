# Darkling Engine

This directory contains a minimal CUDA-based cracking kernel used for low-bandwidth workers.
It performs mask-based password generation and hashing entirely on the GPU with very little
PCIe traffic.

The `darkling_engine.cu` file exposes a `launch_darkling` function that accepts a
start and end counter. Up to sixteen custom charsets can be defined and mapped
to individual mask positions (e.g. `?1?2?2?3`). The kernel stores UTF‑8 encoded
characters in constant memory for fast lookup and supports non‑ASCII inputs.
A small host helper `darkling_host.cpp` demonstrates how to preload this data
and reuse result buffers across batches.

Compile the kernel with a command similar to:

```
nvcc -O2 -arch=sm_52 -cubin darkling_engine.cu -o darkling-engine.cubin
```

The resulting module can be loaded by a host program that manages device memory and
launch parameters. The companion `darkling_host.cpp` file can be compiled with

```
g++ darkling_host.cpp -o darkling-host -lcuda -lcudart
```

It preloads the kernel data into GPU memory and invokes `launch_darkling` across
multiple counter ranges without re-allocating buffers.

Beginning with version 2 the host code caches charsets and hash lists on the
GPU. `DarklingContext` tracks the previously loaded data and only uploads new
values when they change. This avoids repeated `cudaMemcpyToSymbol` calls and
reduces kernel launch overhead when processing many small ranges.

The `launcher.py` script wraps the binary and selects built-in alphabets
for convenience. Use `--lang` to map `?1`/`?2` to a specific language:

```
python launcher.py --lang German --start 0 --end 1000
```

## Tuning

When run as part of the worker each GPU receives its own darkling instance.
Optional environment variables control power limits and autotuning:

- `DARKLING_GPU_POWER_LIMIT` – apply a static power cap before launching the
  kernel.
- `DARKLING_TARGET_POWER_LIMIT` – attempt to reduce grid size if power draw
  exceeds this threshold.
- `DARKLING_AUTOTUNE` – when set, grid and block sizes are chosen based on
  device properties using `cudaOccupancyMaxActiveBlocksPerMultiprocessor`. The
  first batch measures hash/s and the configuration is adjusted automatically.

These variables can be used in combination with the worker's per-GPU sidecars to
balance performance across multiple devices.

## Multi-Vendor Backends

Darkling now exposes an abstract GPU interface in `gpu_backend.h` with concrete
implementations for CUDA, HIP and OpenCL devices. The dispatcher
`backend_dispatcher.cpp` selects the appropriate backend at runtime or via the
`DARKLING_GPU_BACKEND` environment variable (`cuda`, `hip`, `opencl`).
Each backend follows the same workflow of `initialize`, `load_data`,
`launch_crack_batch` and result polling. Only the CUDA version contains a
working kernel at the moment; the other backends provide build stubs for future
expansion.

## Predefined Charsets

Several common alphabets and symbol sets are bundled in `charsets.py`.
Import the desired constants when preparing mask attacks:

```python
from darkling.charsets import (
    GERMAN_UPPER,
    GERMAN_LOWER,
    EMOJI,
    CHINESE,
    JAPANESE,
)

mask_charsets = {
    '?1': GERMAN_UPPER,
    '?2': GERMAN_LOWER,
    '?3': EMOJI,
    '?4': CHINESE,
    '?5': JAPANESE,
}
```

`charsets.py` exposes uppercase and lowercase alphabets for the top 20
languages along with `COMMON_SYMBOLS`, `EMOJI`, and scripts such as
`CHINESE` and `JAPANESE` for convenience.

## Building the GPU Backends

The CUDA implementation is built by default but HIP and OpenCL backends can
also be compiled via CMake:

```bash
cd darkling
cmake -DENABLE_HIP=ON -DENABLE_OPENCL=ON -B build
cmake --build build
```

`ENABLE_CUDA`, `ENABLE_HIP`, and `ENABLE_OPENCL` may be toggled to target a
specific platform.  The resulting static libraries `cuda_backend`,
`hip_backend`, and `opencl_backend` expose a `launch_darkling` entry point for
each vendor.
