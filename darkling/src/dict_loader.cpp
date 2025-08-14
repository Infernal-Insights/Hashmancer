#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "darkling_dict.h"

static bool dl_map_shard_buffer(uint8_t* data, size_t size, bool external, DlShardView* out) {
  if (size < sizeof(uint32_t)) return false;
  uint32_t count = ((uint32_t*)data)[0];
  size_t header = sizeof(uint32_t) + (count + 1) * sizeof(uint32_t);
  if (size < header) return false;
  uint32_t* offsets = (uint32_t*)(data + sizeof(uint32_t));
  uint32_t sentinel = offsets[count];
  if (header + sentinel > size) return false;
  cudaError_t err = cudaHostRegister(data, size, cudaHostRegisterReadOnly);
  out->base = data;
  out->offsets = offsets;
  out->count = count;
  out->words_offset = header;
  out->bytes = size;
  out->external = external;
  out->registered = (err == cudaSuccess);
  return true;
}

bool dl_map_shard(const char* path, DlShardView* out) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) return false;
  struct stat st;
  if (fstat(fd, &st) != 0) { close(fd); return false; }
  size_t size = st.st_size;
  uint8_t* data = (uint8_t*)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (data == MAP_FAILED) return false;
  if (!dl_map_shard_buffer(data, size, false, out)) {
    munmap(data, size);
    return false;
  }
  return true;
}

bool dl_map_shard_mem(const uint8_t* data, size_t size, DlShardView* out) {
  return dl_map_shard_buffer((uint8_t*)data, size, true, out);
}

void dl_unmap_shard(DlShardView* v) {
  if (!v->base) return;
  if (v->registered) cudaHostUnregister((void*)v->base);
  if (!v->external) {
    munmap((void*)v->base, v->bytes);
  }
  v->base = NULL;
}

