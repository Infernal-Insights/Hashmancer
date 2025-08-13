#pragma once
#include <stdint.h>

enum DlAlgo { DL_MD5=0, DL_SHA1=1, DL_NTLM=2 };
enum DlRuleShape : uint8_t { PREFIX_1, SUFFIX_D4, CASE_TOGGLE, LEET_LIGHT, SUFFIX_SHORT, AFFIX_PAIR };

struct DlRuleMC {
  uint8_t  shape;
  uint8_t  max_len;
  uint16_t flags;
  uint8_t  params[16];
};

struct RuleParams { uint8_t bytes[16]; };

typedef void (*RuleFn)(uint8_t* dst, const uint8_t* src, uint32_t len,
                       const RuleParams* params, uint32_t variant_idx, uint32_t variant_count);

struct DlRuleDispatch {
  DlRuleShape shape;
  RuleFn      fn;
  uint32_t    variants;
};

void dl_rules_load_json(const char* json_path);

extern __device__ __constant__ DlRuleMC g_rules[256];
extern __device__ __constant__ DlRuleDispatch g_dispatch[6];
