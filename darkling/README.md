# Darkling Engine

This directory contains a minimal CUDA-based cracking kernel used for low-bandwidth workers.
It performs mask-based password generation and hashing entirely on the GPU with very little
PCIe traffic.

The `darkling_engine.cu` file exposes a single `launch_darkling` function that copies
charsets and target hashes into constant memory and then launches `crack_kernel`.
Cracked passwords are written to a device buffer and can be retrieved by the caller
when the kernel finishes.

Compile the kernel with a command similar to:

```
nvcc -O2 -arch=sm_52 -cubin darkling_engine.cu -o darkling-engine.cubin
```

The resulting module can be loaded by a host program that manages device memory and
launch parameters.
