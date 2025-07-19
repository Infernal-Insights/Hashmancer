# Darkling Engine

This directory contains a minimal CUDA-based cracking kernel used for low-bandwidth workers.
It performs mask-based password generation and hashing entirely on the GPU with very little
PCIe traffic.

The `darkling_engine.cu` file exposes a `launch_darkling` function that accepts a
start and end counter. Charsets and target hashes are copied into constant memory
only once and remain resident between launches. A small host helper `darkling_host.cpp`
demonstrates how to preload this data and reuse result buffers across batches.

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
