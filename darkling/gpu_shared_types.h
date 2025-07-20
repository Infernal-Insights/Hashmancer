#pragma once
#include <cstdint>
#include <vector>
#include <array>

// Max limits
#define MAX_CHARSETS 16
#define MAX_MASK_LEN 32
#define MAX_HASHES 4096
#define MAX_RESULT_BUFFER 512

// Job Configuration (passed to GPU)
struct MaskJob {
  uint64_t start_index;
  uint64_t end_index;
  uint8_t mask_length;
  uint8_t mask_template[MAX_MASK_LEN];   // charset IDs per position (e.g., ?1?2?2?3 -> [1,2,2,3])
  uint8_t charset_lengths[MAX_CHARSETS];
  uint8_t charsets[MAX_CHARSETS][256];   // flattened LUTs (ASCII or binary)
  uint8_t hash_type;                     // e.g., 1 = MD5, 2 = SHA1
  uint8_t hash_length;
  uint8_t hashes[MAX_HASHES][32];        // variable-length hash buffer (e.g., 16 for MD5)
  uint32_t num_hashes;
};

// GPU result record
struct CrackResult {
  uint64_t candidate_index;
  uint8_t hash[32];      // Matched hash
  uint8_t password[32];  // Cracked plaintext
  uint8_t length;
};

// GPU status block
struct GpuStatus {
  uint64_t hashes_processed;
  float gpu_temp_c;
  float batch_duration_ms;
  bool overheat_flag;
};
