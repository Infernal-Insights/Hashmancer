#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "darkling_dict.h"

bool dl_map_shard(const char* path, DlShardView* out) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  size_t size = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t* data = (uint8_t*)malloc(size);
  if (fread(data,1,size,f)!=size) { fclose(f); free(data); return false; }
  fclose(f);
  uint32_t count = ((uint32_t*)data)[0];
  out->base = data;
  out->offsets = (uint32_t*)(data + sizeof(uint32_t));
  out->count = count;
  out->words_offset = sizeof(uint32_t) + count*sizeof(uint32_t);
  return true;
}

void dl_unmap_shard(DlShardView* v) {
  free((void*)v->base);
  v->base = NULL;
}
