#pragma once
#include <stdint.h>
#include <stddef.h>
struct DlShardView {
  const uint8_t* base;
  const uint32_t* offsets;
  uint64_t count;
  uint32_t words_offset;
  size_t bytes;
  bool external;
  bool registered;
};
bool  dl_map_shard(const char* path, DlShardView* out);
bool  dl_map_shard_mem(const uint8_t* data, size_t bytes, DlShardView* out);
void  dl_unmap_shard(DlShardView*);

struct DlPinnedRing;
DlPinnedRing* dl_ring_create(size_t buf_size_mb, int buffers);
void dl_ring_destroy(DlPinnedRing*);
uint8_t* dl_ring_acquire(DlPinnedRing*, size_t* bytes);
void dl_ring_commit(DlPinnedRing*, size_t bytes);
