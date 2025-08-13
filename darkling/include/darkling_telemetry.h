#pragma once
#include <stdint.h>
struct DlTelemetry {
  uint64_t words_processed;
  uint64_t candidates_generated;
  uint64_t kernel_ms;
  uint64_t h2d_bytes;
  uint64_t d2h_bytes;
  uint64_t hits;
};
