#include <cuda_runtime.h>
#include "darkling_device_queue.h"

__device__ DlWorkItem g_queue[DL_QUEUE_SIZE];
__device__ uint32_t g_qhead = 0;
__device__ uint32_t g_qtail = 0;
__device__ bool g_stop = false;

__device__ void dq_push(const DlWorkItem& item) {
  uint32_t tail = atomicAdd(&g_qtail, 1);
  uint32_t head = g_qhead;
  
  // Check for queue overflow (allow some slack for concurrent operations)
  if (tail - head >= DL_QUEUE_SIZE - 64) {
    atomicSub(&g_qtail, 1);
    return; // Queue full, drop item
  }
  
  uint32_t idx = tail & DL_QUEUE_MASK;
  g_queue[idx] = item;
}

__device__ bool dq_pop(DlWorkItem* out) {
  uint32_t head = g_qhead;
  uint32_t tail = g_qtail;
  if (head >= tail) {
    return false;
  }
  uint32_t expected = head;
  if (atomicCAS(&g_qhead, expected, head + 1) != expected) {
    return false;
  }
  *out = g_queue[head & DL_QUEUE_MASK];
  return true;
}

__device__ uint32_t dq_size() {
  return g_qtail - g_qhead;
}

__device__ bool dq_is_full() {
  return dq_size() >= (DL_QUEUE_SIZE - 64);
}
