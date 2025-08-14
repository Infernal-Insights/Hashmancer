#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include "darkling_rules.h"
#include "darkling_device_queue.h"
#include "darkling_telemetry.h"

void launch_persistent(const uint8_t*, const uint32_t*, DlTelemetry*);

extern __device__ __constant__ uint32_t g_target_hash[5];
extern __device__ __constant__ bool g_target_set;

int main(int argc, char** argv) {
  dl_rules_load_json("configs/ruleset_baked256.json");
  
  uint32_t target_hash[5] = {0x5d41402a, 0xbc4b2a76, 0xb9719d91, 0x1017c592, 0}; // MD5("hello")
  bool target_enabled = true;
  cudaMemcpyToSymbol(g_target_hash, target_hash, sizeof(target_hash));
  cudaMemcpyToSymbol(g_target_set, &target_enabled, sizeof(bool));
  
  const char* words = "hello\nworld\n";
  uint32_t offsets_host[3] = {0,6,12};
  uint8_t* d_words; uint32_t* d_offsets;
  cudaMalloc(&d_words, 12);
  cudaMemcpy(d_words, words, 12, cudaMemcpyHostToDevice);
  cudaMalloc(&d_offsets, sizeof(offsets_host));
  cudaMemcpy(d_offsets, offsets_host, sizeof(offsets_host), cudaMemcpyHostToDevice);

  DlTelemetry* d_tel; cudaMalloc(&d_tel, sizeof(DlTelemetry)); cudaMemset(d_tel,0,sizeof(DlTelemetry));
  DlWorkItem item{0,2,0,2};
  auto start = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - start < std::chrono::seconds(10)) {
    uint32_t zero = 0, tail = 1;
    cudaMemcpyToSymbol(g_qhead, &zero, sizeof(uint32_t));
    cudaMemcpyToSymbol(g_qtail, &zero, sizeof(uint32_t));
    cudaMemcpyToSymbol(g_queue, &item, sizeof(DlWorkItem));
    cudaMemcpyToSymbol(g_qtail, &tail, sizeof(uint32_t));
    launch_persistent(d_words, d_offsets, d_tel);
    cudaDeviceSynchronize();
  }
  DlTelemetry h_tel{};
  cudaMemcpy(&h_tel, d_tel, sizeof(DlTelemetry), cudaMemcpyDeviceToHost);
  printf("words_processed=%llu candidates=%llu hits=%llu queue_max=%llu\n", (unsigned long long)h_tel.words_processed, (unsigned long long)h_tel.candidates_generated, (unsigned long long)h_tel.hits, (unsigned long long)h_tel.queue_max_size);
  cudaFree(d_tel); cudaFree(d_words); cudaFree(d_offsets);
  return 0;
}
