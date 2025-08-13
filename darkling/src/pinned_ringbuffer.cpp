#include <cuda_runtime.h>
#include <stdlib.h>
#include "darkling_dict.h"

struct DlPinnedRing {
  size_t buf_size;
  int buffers;
  uint8_t** ptrs;
  int head;
};

DlPinnedRing* dl_ring_create(size_t buf_size_mb, int buffers) {
  DlPinnedRing* r = (DlPinnedRing*)malloc(sizeof(DlPinnedRing));
  r->buf_size = buf_size_mb * 1024 * 1024;
  r->buffers = buffers;
  r->ptrs = (uint8_t**)malloc(sizeof(uint8_t*) * buffers);
  for (int i = 0; i < buffers; ++i) cudaHostAlloc(&r->ptrs[i], r->buf_size, cudaHostAllocDefault);
  r->head = 0;
  return r;
}

void dl_ring_destroy(DlPinnedRing* r) {
  for (int i = 0; i < r->buffers; ++i) cudaFreeHost(r->ptrs[i]);
  free(r->ptrs);
  free(r);
}

uint8_t* dl_ring_acquire(DlPinnedRing* r, size_t* bytes) {
  *bytes = r->buf_size;
  return r->ptrs[r->head];
}

void dl_ring_commit(DlPinnedRing* r, size_t bytes) {
  r->head = (r->head + 1) % r->buffers;
}
