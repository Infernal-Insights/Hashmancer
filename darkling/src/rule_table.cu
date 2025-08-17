#include <cuda_runtime.h>
#include <string.h>
#include "darkling_rules.h"

extern "C" __device__ uint32_t rule_prefix_1(uint8_t*, const uint8_t*, uint32_t,
                                             const RuleParams*, uint32_t, uint32_t);
extern "C" __device__ uint32_t rule_suffix_d4(uint8_t*, const uint8_t*, uint32_t,
                                               const RuleParams*, uint32_t, uint32_t);

__device__ __constant__ DlRuleMC g_rules[256];
__device__ __constant__ DlRuleDispatch g_dispatch[6];

void dl_rules_load_json(const char* path) {
  DlRuleMC host_rules[256];
  memset(host_rules, 0, sizeof(host_rules));
  host_rules[0].shape = PREFIX_1;
  host_rules[0].max_len = 32;
  host_rules[0].length_delta = 1;  // Adds 1 character
  host_rules[0].params[0] = 0;
  host_rules[1].shape = SUFFIX_D4;
  host_rules[1].max_len = 32;
  host_rules[1].length_delta = 4;  // Adds 4 digits
  cudaMemcpyToSymbol(g_rules, host_rules, sizeof(host_rules));

  DlRuleDispatch disp[6];
  memset(disp, 0, sizeof(disp));
  disp[PREFIX_1].shape = PREFIX_1;
  disp[PREFIX_1].fn = rule_prefix_1;
  disp[PREFIX_1].variants = 14;
  disp[SUFFIX_D4].shape = SUFFIX_D4;
  disp[SUFFIX_D4].fn = rule_suffix_d4;
  disp[SUFFIX_D4].variants = 10000;
  cudaMemcpyToSymbol(g_dispatch, disp, sizeof(disp));
}
