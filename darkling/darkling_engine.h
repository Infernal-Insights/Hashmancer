#ifndef DARKLING_ENGINE_H
#define DARKLING_ENGINE_H
#include <stdint.h>

#define MAX_MASK_LEN 32
#define MAX_CHARSET_SIZE 128
#define MAX_HASHES 1024

#ifdef __cplusplus
extern "C" {
#endif

extern __constant__ char d_charsets[MAX_MASK_LEN][MAX_CHARSET_SIZE];
extern __constant__ int  d_charset_lens[MAX_MASK_LEN];
extern __constant__ uint8_t d_hashes[MAX_HASHES][20];
extern __constant__ int d_num_hashes;
extern __constant__ int d_hash_len;
extern __constant__ int d_pwd_len;

void launch_darkling(const char **charsets, const int *lens, int pwd_len,
                     const uint8_t *hashes, int num_hashes, int hash_len,
                     uint64_t start, uint64_t end,
                     char *d_results, int max_results, int *d_count,
                     dim3 grid, dim3 block);

#ifdef __cplusplus
}
#endif

#endif // DARKLING_ENGINE_H
