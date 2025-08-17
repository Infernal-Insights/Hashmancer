#include <cuda_runtime.h>
#include <stdio.h>
#include "darkling_rules.h"
#include "darkling_device_queue.h"
#include "darkling_telemetry.h"
#include "hash_primitives.cuh"
#include "kernel_optimizations.cuh"

#ifdef DARK_ENABLE_PTX_RULES
// PTX rule function declarations
extern "C" __device__ void rule_prefix_1_ptx(uint8_t*, const uint8_t*, uint32_t, const RuleParams*, uint32_t, uint32_t);
extern "C" __device__ void rule_suffix_d4_ptx(uint8_t*, const uint8_t*, uint32_t, const RuleParams*, uint32_t, uint32_t);
extern "C" __device__ void rule_vectorized_copy_ptx(uint8_t*, const uint8_t*, uint32_t, const RuleParams*, uint32_t, uint32_t);
extern "C" __device__ void rule_case_toggle_simd_ptx(uint8_t*, const uint8_t*, uint32_t, const RuleParams*, uint32_t, uint32_t);
extern "C" __device__ void rule_leet_lookup_optimized_ptx(uint8_t*, const uint8_t*, uint32_t, const RuleParams*, uint32_t, uint32_t);
#endif

extern "C" __device__ uint32_t rule_prefix_1(uint8_t* dst, const uint8_t* src, uint32_t len,
                                             const RuleParams* params, uint32_t variant_idx, uint32_t variant_count);
extern "C" __device__ uint32_t rule_suffix_d4(uint8_t* dst, const uint8_t* src, uint32_t len,
                                               const RuleParams* params, uint32_t variant_idx, uint32_t variant_count);

__global__ void persistent_kernel(const uint8_t* words, const uint32_t* offsets, DlTelemetry* tel) {
  DlWorkItem item;
  uint32_t max_queue_size = 0;
  while (dq_pop(&item)) {
    uint32_t current_size = dq_size();
    if (current_size > max_queue_size) {
      max_queue_size = current_size;
      atomicMax((unsigned long long*)&tel->queue_max_size, (unsigned long long)current_size);
    }
    if (item.word_count == 0) continue;
    uint32_t end = offsets[item.word_start + item.word_count];
    uint32_t last = offsets[item.word_start + item.word_count - 1];
    if (end <= last) {
      printf("missing sentinel for word range %u-%u\n", item.word_start, item.word_start + item.word_count);
      continue;
    }
    for (uint32_t i = 0; i < item.word_count; ++i) {
      uint32_t idx = item.word_start + i;
      const uint8_t* src = words + offsets[idx];
      uint32_t len = offsets[idx+1] - offsets[idx] - 1; // strip \n
      DlRuleMC rule = g_rules[item.rule_start];
      RuleParams params;
      #pragma unroll
      for (int j = 0; j < 16; ++j) params.bytes[j] = rule.params[j];
      DlRuleDispatch disp = g_dispatch[rule.shape];
      uint8_t tmp[128];
      for (uint32_t v = 0; v < 1; ++v) {
        uint32_t candidate_len = disp.fn(tmp, src, len, &params, v, disp.variants);
        if (candidate_len > 120) continue;
        uint32_t dig[4];
        md5_hash(tmp, candidate_len, dig);
        atomicAdd(&tel->candidates_generated, 1ULL);
        if (check_hash(dig)) {
          atomicAdd(&tel->hits, 1ULL);
        }
      }
      atomicAdd(&tel->words_processed, 1ULL);
    }
  }
}

void launch_persistent(const uint8_t* d_words, const uint32_t* d_offsets, DlTelemetry* tel) {
  int device;
  cudaGetDevice(&device);
  
  // Use optimized launch parameters
  KernelOptParams opt_params = get_optimal_launch_params(device);
  
  // Configure shared memory for optimal performance
  cudaFuncSetAttribute(persistent_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, opt_params.shared_mem_bytes);
  
  // Launch with optimized parameters
  persistent_kernel<<<opt_params.blocks, opt_params.threads_per_block, opt_params.shared_mem_bytes>>>(d_words, d_offsets, tel);
}
