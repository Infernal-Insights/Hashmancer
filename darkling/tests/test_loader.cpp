#include <assert.h>
#include <stdio.h>
#include <string.h>
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
  return 0;
}
