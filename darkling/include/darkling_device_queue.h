#pragma once
#include <stdint.h>

struct DlWorkItem {
  uint32_t word_start;
  uint32_t word_count;
  uint32_t rule_start;
  uint32_t rule_count;
};

__device__ void dq_push(const DlWorkItem&);
__device__ bool dq_pop(DlWorkItem*);

extern __device__ DlWorkItem g_queue[64];
extern __device__ uint32_t g_qhead;
extern __device__ uint32_t g_qtail;
