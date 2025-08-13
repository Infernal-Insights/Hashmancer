#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "darkling_dict.h"

int main() {
  const char* path = "darkling_test.bin";
  FILE* f = fopen(path, "wb");
  uint32_t count = 2;
  uint32_t offsets[3] = {0,6,12};
  fwrite(&count,4,1,f);
  fwrite(offsets,4,3,f);
  fwrite("hello\nworld\n",1,12,f);
  fclose(f);
  DlShardView v; bool ok = dl_map_shard(path,&v); assert(ok); assert(v.count==2);
  dl_unmap_shard(&v);
  remove(path);

  size_t mem_size = 4 + 12 + 12;
  uint8_t* mem = (uint8_t*)malloc(mem_size);
  memcpy(mem, &count, 4);
  memcpy(mem + 4, offsets, 12);
  memcpy(mem + 16, "hello\nworld\n", 12);
  DlShardView v2; bool ok2 = dl_map_shard_mem(mem, mem_size, &v2); assert(ok2); assert(v2.count==2);
  dl_unmap_shard(&v2);
  free(mem);
  return 0;
}
