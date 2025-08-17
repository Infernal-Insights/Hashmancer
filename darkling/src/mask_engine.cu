#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>
#include "darkling_mask_engine.h"
#include "darkling_telemetry.h"
#include "hash_primitives.cuh"
#include "kernel_optimizations.cuh"

// GPU constant memory for charset data
__device__ __constant__ uint8_t g_mask_charsets[8192];  // 8KB charset data
__device__ __constant__ DlMaskPos g_mask_positions[32]; // Position descriptors
__device__ __constant__ uint32_t g_mask_length;

// Predefined character sets
static const char* CHARSET_LOWER = "abcdefghijklmnopqrstuvwxyz";
static const char* CHARSET_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
static const char* CHARSET_DIGIT = "0123456789";
static const char* CHARSET_SPECIAL = "!@#$%^&*()-_=+[]{}|;:,.<>?";
static const char* CHARSET_HEX_LOWER = "0123456789abcdef";
static const char* CHARSET_HEX_UPPER = "0123456789ABCDEF";

// GPU mask candidate generation kernel
__global__ void mask_generate_kernel(
    uint8_t* candidates,        // [batch_size * max_length]
    uint32_t* lengths,          // [batch_size] 
    uint64_t start_index,
    uint32_t batch_size,
    uint32_t max_length) {
    
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    
    for (uint32_t i = tid; i < batch_size; i += stride) {
        uint64_t candidate_index = start_index + i;
        uint8_t* candidate = candidates + i * max_length;
        
        // Decode candidate index into character positions
        uint64_t remaining = candidate_index;
        uint32_t pos = 0;
        
        for (uint32_t p = 0; p < g_mask_length; ++p) {
            DlMaskPos mask_pos = g_mask_positions[p];
            
            if (mask_pos.type == MASK_LITERAL) {
                // Literal character - just copy it
                candidate[pos++] = g_mask_charsets[mask_pos.start_offset];
            } else {
                // Variable character - decode from index
                uint32_t char_index = remaining % mask_pos.char_count;
                remaining /= mask_pos.char_count;
                candidate[pos++] = g_mask_charsets[mask_pos.start_offset + char_index];
            }
        }
        
        lengths[i] = pos;
    }
}

// Optimized mask generation with vectorization
__global__ void mask_generate_vectorized_kernel(
    uint8_t* candidates,
    uint32_t* lengths, 
    uint64_t start_index,
    uint32_t batch_size,
    uint32_t max_length) {
    
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    
    // Cooperative warp-level generation
    if (warp_id < (batch_size + 31) / 32) {
        uint32_t candidates_per_warp = min(32u, batch_size - warp_id * 32);
        
        for (uint32_t w = 0; w < candidates_per_warp; ++w) {
            uint32_t candidate_id = warp_id * 32 + w;
            uint64_t candidate_index = start_index + candidate_id;
            
            if (lane_id < g_mask_length) {
                // Each thread handles one position
                DlMaskPos mask_pos = g_mask_positions[lane_id];
                uint64_t pos_index = candidate_index;
                
                // Calculate position index
                for (uint32_t p = 0; p < lane_id; ++p) {
                    pos_index /= g_mask_positions[p].char_count;
                }
                
                uint8_t char_val;
                if (mask_pos.type == MASK_LITERAL) {
                    char_val = g_mask_charsets[mask_pos.start_offset];
                } else {
                    uint32_t char_index = pos_index % mask_pos.char_count;
                    char_val = g_mask_charsets[mask_pos.start_offset + char_index];
                }
                
                // Store character
                uint8_t* candidate = candidates + candidate_id * max_length;
                candidate[lane_id] = char_val;
            }
            
            // Set length (first thread in warp)
            if (lane_id == 0) {
                lengths[candidate_id] = g_mask_length;
            }
        }
    }
}

// Host implementation
DlMaskCompileResult dl_mask_compile(const char* mask_string) {
    DlMaskCompileResult result = {};
    result.success = false;
    
    if (!mask_string || strlen(mask_string) == 0) {
        strcpy(result.error_msg, "Empty mask string");
        return result;
    }
    
    const char* p = mask_string;
    uint32_t pos_count = 0;
    uint32_t charset_offset = 0;
    uint8_t charset_buffer[8192];
    
    while (*p && pos_count < 32) {
        DlMaskPos& pos = result.mask.positions[pos_count];
        
        if (*p == '?' && *(p + 1)) {
            // Mask character
            char mask_type = *(p + 1);
            pos.type = mask_type;
            pos.start_offset = charset_offset;
            
            const char* charset = nullptr;
            uint32_t charset_len = 0;
            
            switch (mask_type) {
                case 'l':
                    charset = CHARSET_LOWER;
                    charset_len = 26;
                    break;
                case 'u':
                    charset = CHARSET_UPPER; 
                    charset_len = 26;
                    break;
                case 'd':
                    charset = CHARSET_DIGIT;
                    charset_len = 10;
                    break;
                case 's':
                    charset = CHARSET_SPECIAL;
                    charset_len = strlen(CHARSET_SPECIAL);
                    break;
                case 'a':
                    // All printable ASCII (33-126)
                    for (uint32_t i = 33; i <= 126; ++i) {
                        charset_buffer[charset_offset + i - 33] = i;
                    }
                    charset_len = 94;
                    break;
                case 'b':
                    // All bytes 0-255
                    for (uint32_t i = 0; i <= 255; ++i) {
                        charset_buffer[charset_offset + i] = i;
                    }
                    charset_len = 256;
                    break;
                case 'h':
                    charset = CHARSET_HEX_LOWER;
                    charset_len = 16;
                    break;
                case 'H':
                    charset = CHARSET_HEX_UPPER;
                    charset_len = 16;
                    break;
                default:
                    snprintf(result.error_msg, sizeof(result.error_msg), 
                            "Unknown mask type: ?%c", mask_type);
                    return result;
            }
            
            if (charset) {
                memcpy(charset_buffer + charset_offset, charset, charset_len);
            }
            
            pos.char_count = charset_len;
            charset_offset += charset_len;
            p += 2; // Skip ?X
            
        } else {
            // Literal character
            pos.type = MASK_LITERAL;
            pos.start_offset = charset_offset;
            pos.char_count = 1;
            charset_buffer[charset_offset++] = *p;
            p++;
        }
        
        pos_count++;
    }
    
    if (*p) {
        strcpy(result.error_msg, "Mask too long (max 32 positions)");
        return result;
    }
    
    // Allocate and copy charset data
    result.mask.length = pos_count;
    result.mask.charset_data_size = charset_offset;
    result.mask.charset_data = (uint8_t*)malloc(charset_offset);
    memcpy(result.mask.charset_data, charset_buffer, charset_offset);
    
    // Calculate maximum candidates
    uint64_t max_candidates = 1;
    for (uint32_t i = 0; i < pos_count; ++i) {
        if (result.mask.positions[i].type != MASK_LITERAL) {
            max_candidates *= result.mask.positions[i].char_count;
        }
    }
    result.mask.max_candidates = max_candidates;
    
    result.success = true;
    return result;
}

void dl_mask_destroy(DlMask* mask) {
    if (mask && mask->charset_data) {
        free(mask->charset_data);
        mask->charset_data = nullptr;
    }
}

bool dl_mask_upload_to_gpu(const DlMask* mask, void** d_mask, void** d_charset_data) {
    if (!mask || mask->charset_data_size > 8192) {
        return false;
    }
    
    // Upload charset data to constant memory
    cudaError_t err = cudaMemcpyToSymbol(g_mask_charsets, mask->charset_data, mask->charset_data_size);
    if (err != cudaSuccess) return false;
    
    // Upload position data to constant memory
    err = cudaMemcpyToSymbol(g_mask_positions, mask->positions, sizeof(DlMaskPos) * mask->length);
    if (err != cudaSuccess) return false;
    
    // Upload mask length
    err = cudaMemcpyToSymbol(g_mask_length, &mask->length, sizeof(uint32_t));
    if (err != cudaSuccess) return false;
    
    *d_mask = (void*)1;  // Dummy pointer since we use constant memory
    *d_charset_data = (void*)1;
    return true;
}

void dl_mask_launch_kernel(
    const void* d_mask,
    const void* d_charset_data, 
    uint8_t* d_candidates,
    uint32_t* d_lengths,
    uint64_t start_index,
    uint32_t batch_size,
    uint32_t max_length) {
    
    // Get optimal launch parameters
    int device;
    cudaGetDevice(&device);
    KernelOptParams params = get_optimal_launch_params(device);
    
    // Choose kernel based on mask characteristics
    bool use_vectorized = (max_length <= 32 && batch_size >= 1024);
    
    if (use_vectorized) {
        mask_generate_vectorized_kernel<<<params.blocks, params.threads_per_block>>>(
            d_candidates, d_lengths, start_index, batch_size, max_length);
    } else {
        mask_generate_kernel<<<params.blocks, params.threads_per_block>>>(
            d_candidates, d_lengths, start_index, batch_size, max_length);
    }
}

uint64_t dl_mask_calculate_keyspace(const DlMask* mask) {
    uint64_t keyspace = 1;
    for (uint32_t i = 0; i < mask->length; ++i) {
        if (mask->positions[i].type != MASK_LITERAL) {
            keyspace *= mask->positions[i].char_count;
        }
    }
    return keyspace;
}

// Hybrid attack kernel combining dictionary words with mask
__global__ void hybrid_dict_mask_kernel(
    const DlDictWord* dict_words,
    const uint8_t* dict_data,
    uint8_t* candidates,
    uint32_t* lengths,
    uint32_t dict_word_index,
    uint64_t mask_start_index,
    uint32_t batch_size,
    uint32_t max_length) {
    
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= batch_size) return;
    
    uint8_t* candidate = candidates + tid * max_length;
    DlDictWord word = dict_words[dict_word_index];
    
    // Copy dictionary word
    const uint8_t* word_data = dict_data + word.offset;
    uint32_t pos = 0;
    for (uint32_t i = 0; i < word.length; ++i) {
        candidate[pos++] = word_data[i];
    }
    
    // Generate mask portion
    uint64_t mask_index = mask_start_index + tid;
    uint64_t remaining = mask_index;
    
    for (uint32_t p = 0; p < g_mask_length; ++p) {
        DlMaskPos mask_pos = g_mask_positions[p];
        
        if (mask_pos.type == MASK_LITERAL) {
            candidate[pos++] = g_mask_charsets[mask_pos.start_offset];
        } else {
            uint32_t char_index = remaining % mask_pos.char_count;
            remaining /= mask_pos.char_count;
            candidate[pos++] = g_mask_charsets[mask_pos.start_offset + char_index];
        }
    }
    
    lengths[tid] = pos;
}

bool dl_mask_estimate_runtime(const DlMask* mask, uint64_t hash_rate, uint64_t* estimated_seconds) {
    uint64_t keyspace = dl_mask_calculate_keyspace(mask);
    if (hash_rate == 0) return false;
    
    *estimated_seconds = keyspace / hash_rate;
    return true;
}