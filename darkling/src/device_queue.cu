#include <cuda_runtime.h>
#include "darkling_device_queue.h"

__device__ DlWorkItem g_queue[64];
__device__ uint32_t g_qhead = 0;
__device__ uint32_t g_qtail = 0;
__device__ bool g_stop = false;

__device__ void dq_push(const DlWorkItem& item) {
  uint32_t idx = atomicAdd(&g_qtail, 1) & 63u;
  g_queue[idx] = item;
}

__device__ bool dq_pop(DlWorkItem* out) {
  uint32_t head = atomicAdd(&g_qhead, 1);
  if (head >= g_qtail) {
    atomicSub(&g_qhead, 1);
    return false;
  }
  *out = g_queue[head & 63u];
  return true;
}
