#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "darkling_rules.h"
#include "darkling_device_queue.h"
#include "darkling_telemetry.h"
#include "hash_primitives.cuh"

// Minimal constants for mask kernel
#define MAX_UTF8_BYTES 4
#define MAX_CUSTOM_SETS 16
#define MAX_CHARSET_CHARS 256
#define MAX_MASK_LEN 32

extern "C" void load_darkling_data(const uint8_t **charset_bytes,
                                    const uint8_t **charset_lens,
                                    const int *charset_sizes,
                                    const uint8_t *pos_map, int pwd_len,
                                    const uint8_t *hashes, int num_hashes, int hash_len,
                                    uint8_t hash_type);

extern "C" void launch_darkling_kernel(uint64_t start, uint64_t end,
                                        char *d_results, int max_results, int *d_count,
                                        dim3 grid, dim3 block);

void launch_persistent(const uint8_t*, const uint32_t*, DlTelemetry*);

// -----------------------------------------------------------------------------
// Dictionary-only kernel used for attack 1
// -----------------------------------------------------------------------------
__global__ void dict_only_kernel(const uint8_t* words, const uint32_t* offsets,
                                 DlTelemetry* tel) {
  DlWorkItem item;
  while (dq_pop(&item)) {
    if (item.word_count == 0) continue;
    for (uint32_t i = 0; i < item.word_count; ++i) {
      uint32_t idx = item.word_start + i;
      const uint8_t* src = words + offsets[idx];
      uint32_t len = offsets[idx+1] - offsets[idx] - 1; // strip newline
      uint32_t dig[4];
      md5_hash(src, len, dig);
      atomicAdd(&tel->candidates_generated, 1ULL);
      atomicAdd(&tel->words_processed, 1ULL);
    }
  }
}

static void launch_dict_only(const uint8_t* d_words, const uint32_t* d_offsets,
                             DlTelemetry* tel) {
  dict_only_kernel<<<1,32>>>(d_words, d_offsets, tel);
}

// -----------------------------------------------------------------------------
// Attack helpers
// -----------------------------------------------------------------------------
static void run_mask_attack(const char* mask) {
  if (!mask) {
    std::fprintf(stderr, "--mask required for attack 0\n");
    return;
  }
  // support numeric masks using built-in digit charset
  int mlen = 0;
  for (const char* p = mask; *p; ++p) {
    if (*p == '?') {
      if (*(p+1)) ++p; // skip charset id
    }
    ++mlen;
  }
  if (mlen <= 0 || mlen > MAX_MASK_LEN) {
    std::fprintf(stderr, "invalid mask length\n");
    return;
  }
  uint8_t digit_bytes[10][MAX_UTF8_BYTES];
  uint8_t digit_lens[10];
  for (int i = 0; i < 10; ++i) {
    digit_bytes[i][0] = '0' + i;
    for (int j = 1; j < MAX_UTF8_BYTES; ++j) digit_bytes[i][j] = 0;
    digit_lens[i] = 1;
  }
  const uint8_t* cs_bytes[MAX_CUSTOM_SETS];
  const uint8_t* cs_lens[MAX_CUSTOM_SETS];
  int cs_sizes[MAX_CUSTOM_SETS] = {0};
  cs_bytes[0] = (const uint8_t*)digit_bytes;
  cs_lens[0] = digit_lens;
  cs_sizes[0] = 10;
  uint8_t pos_map[MAX_MASK_LEN];
  for (int i = 0; i < mlen; ++i) pos_map[i] = 0;
  uint8_t dummy_hash[16] = {0};
  char* d_results; int* d_count;
  cudaMalloc(&d_results, 1);
  cudaMalloc(&d_count, sizeof(int));
  load_darkling_data(cs_bytes, cs_lens, cs_sizes, pos_map, mlen,
                     dummy_hash, 0, 16, 1);
  uint64_t end = 1; for (int i = 0; i < mlen; ++i) end *= 10ULL;
  dim3 grid(1), block(1);
  launch_darkling_kernel(0, end, d_results, 0, d_count, grid, block);
  cudaDeviceSynchronize();
  cudaFree(d_results); cudaFree(d_count);
  std::printf("mask attack generated %llu candidates\n",
              (unsigned long long)end);
}

static void run_dict_only(const char* shard) {
  (void)shard; // shard loading not yet implemented
  const char* words = "hello\nworld\n";
  uint32_t offsets_host[3] = {0,6,12};
  uint8_t* d_words; uint32_t* d_offsets;
  cudaMalloc(&d_words, 12);
  cudaMemcpy(d_words, words, 12, cudaMemcpyHostToDevice);
  cudaMalloc(&d_offsets, sizeof(offsets_host));
  cudaMemcpy(d_offsets, offsets_host, sizeof(offsets_host), cudaMemcpyHostToDevice);
  DlTelemetry* d_tel; cudaMalloc(&d_tel, sizeof(DlTelemetry));
  cudaMemset(d_tel, 0, sizeof(DlTelemetry));
  DlWorkItem item{0,2,0,0};
  uint32_t zero = 0, tail = 1;
  cudaMemcpyToSymbol(g_qhead, &zero, sizeof(uint32_t));
  cudaMemcpyToSymbol(g_qtail, &zero, sizeof(uint32_t));
  cudaMemcpyToSymbol(g_queue, &item, sizeof(DlWorkItem));
  cudaMemcpyToSymbol(g_qtail, &tail, sizeof(uint32_t));
  launch_dict_only(d_words, d_offsets, d_tel);
  cudaDeviceSynchronize();
  DlTelemetry h{}; cudaMemcpy(&h, d_tel, sizeof(h), cudaMemcpyDeviceToHost);
  std::printf("words_processed=%llu candidates=%llu\n",
              (unsigned long long)h.words_processed,
              (unsigned long long)h.candidates_generated);
  cudaFree(d_tel); cudaFree(d_words); cudaFree(d_offsets);
}

static void run_dict_rules(const char* shard, const char* ruleset) {
  (void)shard; // shard loading not yet implemented
  dl_rules_load_json(ruleset);
  const char* words = "hello\nworld\n";
  uint32_t offsets_host[3] = {0,6,12};
  uint8_t* d_words; uint32_t* d_offsets;
  cudaMalloc(&d_words, 12);
  cudaMemcpy(d_words, words, 12, cudaMemcpyHostToDevice);
  cudaMalloc(&d_offsets, sizeof(offsets_host));
  cudaMemcpy(d_offsets, offsets_host, sizeof(offsets_host), cudaMemcpyHostToDevice);
  DlTelemetry* d_tel; cudaMalloc(&d_tel, sizeof(DlTelemetry));
  cudaMemset(d_tel, 0, sizeof(DlTelemetry));
  DlWorkItem item{0,2,0,2};
  uint32_t zero = 0, tail = 1;
  cudaMemcpyToSymbol(g_qhead, &zero, sizeof(uint32_t));
  cudaMemcpyToSymbol(g_qtail, &zero, sizeof(uint32_t));
  cudaMemcpyToSymbol(g_queue, &item, sizeof(DlWorkItem));
  cudaMemcpyToSymbol(g_qtail, &tail, sizeof(uint32_t));
  launch_persistent(d_words, d_offsets, d_tel);
  cudaDeviceSynchronize();
  DlTelemetry h{}; cudaMemcpy(&h, d_tel, sizeof(h), cudaMemcpyDeviceToHost);
  std::printf("words_processed=%llu candidates=%llu\n",
              (unsigned long long)h.words_processed,
              (unsigned long long)h.candidates_generated);
  cudaFree(d_tel); cudaFree(d_words); cudaFree(d_offsets);
}

static void run_external_rules(const char* shard, const char* rulefile) {
  (void)shard; (void)rulefile;
  std::fprintf(stderr, "attack 3 (external rules) not implemented\n");
}

int main(int argc, char** argv) {
  int attack = 0;
  const char* shard = nullptr;
  const char* ruleset = "configs/ruleset_baked256.json";
  const char* mask = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--attack") == 0 && i + 1 < argc) {
      attack = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--shard") == 0 && i + 1 < argc) {
      shard = argv[++i];
    } else if (std::strcmp(argv[i], "--ruleset") == 0 && i + 1 < argc) {
      ruleset = argv[++i];
    } else if (std::strcmp(argv[i], "--mask") == 0 && i + 1 < argc) {
      mask = argv[++i];
    }
  }

  switch (attack) {
    case 0: run_mask_attack(mask); break;
    case 1: run_dict_only(shard); break;
    case 2: run_dict_rules(shard, ruleset); break;
    case 3: run_external_rules(shard, ruleset); break;
    default:
      std::fprintf(stderr, "unknown attack mode %d\n", attack);
      return 1;
  }
  return 0;
}
