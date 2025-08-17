#include <assert.h>
#include <cuda_runtime.h>
#include "darkling_rules.h"
#include "darkling_device_queue.h"
#include "darkling_telemetry.h"

void launch_persistent(const uint8_t*, const uint32_t*, DlTelemetry*);

int main() {
  dl_rules_load_json("configs/ruleset_baked256.json");
  const char* word = "test\n";
  uint32_t offsets[2] = {0,5};
  uint8_t* d_words; uint32_t* d_offsets; DlTelemetry* tel;
  cudaMalloc(&d_words,5); cudaMemcpy(d_words,word,5,cudaMemcpyHostToDevice);
  cudaMalloc(&d_offsets,sizeof(offsets)); cudaMemcpy(d_offsets,offsets,sizeof(offsets),cudaMemcpyHostToDevice);
  cudaMalloc(&tel,sizeof(DlTelemetry)); cudaMemset(tel,0,sizeof(DlTelemetry));
  DlWorkItem item{0,1,0,1};
  uint32_t zero=0, tail=1;
  cudaMemcpyToSymbol(g_qhead,&zero,sizeof(uint32_t));
  cudaMemcpyToSymbol(g_qtail,&zero,sizeof(uint32_t));
  cudaMemcpyToSymbol(g_queue,&item,sizeof(DlWorkItem));
  cudaMemcpyToSymbol(g_qtail,&tail,sizeof(uint32_t));
  launch_persistent(d_words,d_offsets,tel);
  cudaDeviceSynchronize();
  DlTelemetry h; cudaMemcpy(&h,tel,sizeof(h),cudaMemcpyDeviceToHost);
  assert(h.words_processed==1);
  cudaFree(tel); cudaFree(d_words); cudaFree(d_offsets);
  return 0;
}
