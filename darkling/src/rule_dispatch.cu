#include <cuda_runtime.h>
#include "darkling_rules.h"

extern "C" __device__ void rule_prefix_1(uint8_t* dst, const uint8_t* src, uint32_t len,
                                         const RuleParams* params, uint32_t variant_idx, uint32_t variant_count) {
  if (len > 126) return;
  static __constant__ uint8_t table[14] = { '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '=', '-' };
  uint8_t sym = table[params->bytes[0] % 14];
  dst[0] = sym;
  for (uint32_t i = 0; i < len; ++i) dst[i+1] = src[i];
}

extern "C" __device__ void rule_suffix_d4(uint8_t* dst, const uint8_t* src, uint32_t len,
                                          const RuleParams* params, uint32_t variant_idx, uint32_t variant_count) {
  if (len > 123) return;
  for (uint32_t i = 0; i < len; ++i) dst[i] = src[i];
  uint32_t v = variant_idx;
  dst[len+0] = '0' + (v % 10);
  dst[len+1] = '0' + ((v/10) % 10);
  dst[len+2] = '0' + ((v/100) % 10);
  dst[len+3] = '0' + ((v/1000) % 10);
}
