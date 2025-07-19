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
