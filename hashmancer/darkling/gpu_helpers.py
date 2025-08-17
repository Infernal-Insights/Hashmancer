from __future__ import annotations

import ctypes.util

try:
    import pyopencl as cl  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cl = None


class GPUContext:
    """Minimal GPU helper supporting OpenCL when available.

    Allocation functions fall back to host memory when GPU libraries are
    unavailable so unit tests can run without hardware.
    """

    def __init__(self) -> None:
        self.ctx = None
        self.queue = None
        if cl is not None:
            try:
                platforms = cl.get_platforms()
                if platforms:
                    self.ctx = cl.Context(devices=[platforms[0].get_devices()[0]])
                    self.queue = cl.CommandQueue(self.ctx)
            except Exception:
                self.ctx = None
                self.queue = None

    def alloc(self, size: int):
        if self.ctx is not None:
            return cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size)
        return bytearray(size)

    def copy_from_host(self, dest, data: bytes) -> None:
        if self.ctx is not None:
            cl.enqueue_copy(self.queue, dest, data)
        else:
            if isinstance(dest, bytearray):
                dest[: len(data)] = data

    def free(self, buf) -> None:  # pragma: no cover - nothing to do for OpenCL
        pass
