#pragma once
#include <stdint.h>
struct DlShardView { const uint8_t* base; const uint32_t* offsets; uint64_t count; uint32_t words_offset; };
bool  dl_map_shard(const char* path, DlShardView* out);
void  dl_unmap_shard(DlShardView*);

struct DlPinnedRing;
DlPinnedRing* dl_ring_create(size_t buf_size_mb, int buffers);
void dl_ring_destroy(DlPinnedRing*);
uint8_t* dl_ring_acquire(DlPinnedRing*, size_t* bytes);
void dl_ring_commit(DlPinnedRing*, size_t bytes);
