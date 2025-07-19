#ifndef DARKLING_ENGINE_H
#define DARKLING_ENGINE_H
#include <stdint.h>

#define MAX_MASK_LEN 32
#define MAX_UTF8_BYTES 4
#define MAX_CUSTOM_SETS 16
#define MAX_CHARSET_CHARS 256
#define MAX_PWD_BYTES (MAX_MASK_LEN * MAX_UTF8_BYTES)
#define MAX_CHARSET_SIZE (MAX_CHARSET_CHARS * MAX_UTF8_BYTES)
#define MAX_HASHES 1024

#ifdef __cplusplus
extern "C" {
#endif

extern __constant__ uint8_t d_charset_bytes[MAX_CUSTOM_SETS][MAX_CHARSET_CHARS][MAX_UTF8_BYTES];
extern __constant__ uint8_t d_charset_charlen[MAX_CUSTOM_SETS][MAX_CHARSET_CHARS];
extern __constant__ int  d_charset_lens[MAX_CUSTOM_SETS];
extern __constant__ uint8_t d_pos_charset[MAX_MASK_LEN];
extern __constant__ uint8_t d_hashes[MAX_HASHES][20];
extern __constant__ int d_num_hashes;
extern __constant__ int d_hash_len;
extern __constant__ int d_pwd_len;

void launch_darkling(const uint8_t **charset_bytes,
                     const uint8_t **charset_lens,
                     const int *charset_sizes,
                     const uint8_t *pos_map, int pwd_len,
                     const uint8_t *hashes, int num_hashes, int hash_len,
                     uint64_t start, uint64_t end,
                     char *d_results, int max_results, int *d_count,
                     dim3 grid, dim3 block);

#ifdef __cplusplus
}
#endif

#endif // DARKLING_ENGINE_H
