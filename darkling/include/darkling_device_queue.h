#pragma once
#include <stdint.h>

#ifndef DL_QUEUE_SIZE
#define DL_QUEUE_SIZE 4096
#endif

#define DL_QUEUE_MASK (DL_QUEUE_SIZE - 1)

struct DlWorkItem {
  uint32_t word_start;
  uint32_t word_count;
  uint32_t rule_start;
  uint32_t rule_count;
};

__device__ void dq_push(const DlWorkItem&);
__device__ bool dq_pop(DlWorkItem*);
__device__ uint32_t dq_size();
__device__ bool dq_is_full();

extern __device__ DlWorkItem g_queue[DL_QUEUE_SIZE];
extern __device__ uint32_t g_qhead;
extern __device__ uint32_t g_qtail;
