#include <cuda_runtime.h>
#include "darkling_rules.h"
#include "darkling_device_queue.h"
#include "darkling_telemetry.h"
#include "hash_primitives.cuh"

extern "C" __device__ void rule_prefix_1(uint8_t* dst, const uint8_t* src, uint32_t len,
                                         const RuleParams* params, uint32_t variant_idx, uint32_t variant_count);
extern "C" __device__ void rule_suffix_d4(uint8_t* dst, const uint8_t* src, uint32_t len,
                                           const RuleParams* params, uint32_t variant_idx, uint32_t variant_count);

__global__ void persistent_kernel(const uint8_t* words, const uint32_t* offsets, DlTelemetry* tel) {
  DlWorkItem item;
  while (dq_pop(&item)) {
    for (uint32_t i = 0; i < item.word_count; ++i) {
      uint32_t idx = item.word_start + i;
      const uint8_t* src = words + offsets[idx];
      uint32_t len = offsets[idx+1] - offsets[idx] - 1; // strip \n
      DlRuleMC rule = g_rules[item.rule_start];
      RuleParams params;
      #pragma unroll
      for (int j = 0; j < 16; ++j) params.bytes[j] = rule.params[j];
      DlRuleDispatch disp = g_dispatch[rule.shape];
      uint8_t tmp[64];
      for (uint32_t v = 0; v < 1; ++v) {
        disp.fn(tmp, src, len, &params, v, disp.variants);
        uint32_t dig[4];
        md5_hash(tmp, len+1, dig);
        atomicAdd(&tel->candidates_generated, 1ULL);
      }
      atomicAdd(&tel->words_processed, 1ULL);
    }
  }
}

void launch_persistent(const uint8_t* d_words, const uint32_t* d_offsets, DlTelemetry* tel) {
  persistent_kernel<<<1,32>>>(d_words, d_offsets, tel);
}
