#pragma once
#include <cuda_runtime.h>

// Optimized kernel launch parameters based on GPU architecture
struct KernelOptParams {
    int blocks;
    int threads_per_block;
    int shared_mem_bytes;
    int max_registers;
};

__host__ KernelOptParams get_optimal_launch_params(int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    
    KernelOptParams params;
    
    // Optimize for different architectures
    if (prop.major >= 8) {  // Ampere/Ada (RTX 30/40 series)
        params.blocks = prop.multiProcessorCount * 4;  // Higher occupancy
        params.threads_per_block = 512;  // Sweet spot for Ampere
        params.shared_mem_bytes = 32768;  // 32KB shared memory
        params.max_registers = 64;
    } else if (prop.major == 7) {  // Turing (RTX 20 series)
        params.blocks = prop.multiProcessorCount * 3;
        params.threads_per_block = 256;
        params.shared_mem_bytes = 16384;  // 16KB shared memory
        params.max_registers = 48;
    } else {  // Older architectures
        params.blocks = prop.multiProcessorCount * 2;
        params.threads_per_block = 128;
        params.shared_mem_bytes = 8192;   // 8KB shared memory
        params.max_registers = 32;
    }
    
    return params;
}

// Memory coalescing helpers
template<typename T>
__device__ __forceinline__ void coalesced_load_4(T* dst, const T* src) {
    // Load 4 elements in a coalesced pattern
    float4* dst4 = reinterpret_cast<float4*>(dst);
    const float4* src4 = reinterpret_cast<const float4*>(src);
    *dst4 = *src4;
}

template<typename T>
__device__ __forceinline__ void coalesced_store_4(T* dst, const T* src) {
    // Store 4 elements in a coalesced pattern
    float4* dst4 = reinterpret_cast<float4*>(dst);
    const float4* src4 = reinterpret_cast<const float4*>(src);
    *dst4 = *src4;
}

// Register-optimized hash state structure
struct __align__(16) HashState {
    uint32_t h[8];  // Aligned for vectorized operations
};

// Optimized warp-level primitives for password generation
class WarpPasswordGen {
private:
    static __device__ __forceinline__ uint32_t warp_reduce_or(uint32_t val) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val |= __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        return val;
    }
    
public:
    // Generate multiple password candidates per warp simultaneously
    static __device__ void generate_batch(
        uint8_t* output_buffer,      // [32 warps * 128 bytes] 
        const uint8_t* base_word,    // Base word
        uint32_t base_len,           // Base word length
        uint32_t rule_variants,      // Number of rule variants
        uint32_t warp_id) {
        
        const int lane_id = threadIdx.x & 31;
        const int warp_offset = warp_id * 32 * 128;  // Each warp gets 32*128 bytes
        
        // Each thread in warp processes different variant
        uint32_t variant_id = lane_id;
        
        if (variant_id < rule_variants) {
            uint8_t* thread_output = output_buffer + warp_offset + lane_id * 128;
            
            // Apply rule transformation (this would call PTX rule)
            // For now, simple prefix example
            thread_output[0] = '!' + (variant_id % 14);
            
            // Copy base word using vectorized operations when possible
            if (base_len >= 16 && ((uintptr_t)base_word & 15) == 0) {
                // Aligned 16-byte copy
                for (int i = 0; i < base_len; i += 16) {
                    if (i + 16 <= base_len) {
                        float4 data = *reinterpret_cast<const float4*>(base_word + i);
                        *reinterpret_cast<float4*>(thread_output + 1 + i) = data;
                    } else {
                        // Handle remaining bytes
                        for (int j = i; j < base_len; ++j) {
                            thread_output[1 + j] = base_word[j];
                        }
                    }
                }
            } else {
                // Fallback to byte copy
                for (int i = 0; i < base_len; ++i) {
                    thread_output[1 + i] = base_word[i];
                }
            }
        }
    }
};

// CUDA Cooperative Groups optimization for hash computation
#if __CUDA_ARCH__ >= 600
#include <cooperative_groups.h>
using namespace cooperative_groups;

__device__ void parallel_hash_batch(
    uint32_t* hash_outputs,      // [32 hashes * 4 words]
    const uint8_t* input_batch,  // [32 inputs * 128 bytes]
    const uint32_t* input_lengths, // [32 lengths]
    uint32_t batch_size) {
    
    auto block = this_thread_block();
    auto warp = tiled_partition<32>(block);
    
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x & 31;
    
    if (warp_id == 0) {
        // Warp 0 handles MD5 hashing
        for (int i = lane_id; i < batch_size; i += 32) {
            const uint8_t* input = input_batch + i * 128;
            uint32_t len = input_lengths[i];
            uint32_t* output = hash_outputs + i * 4;
            
            // Call optimized MD5 (would be actual implementation)
            // md5_hash_optimized(input, len, output);
        }
    }
    
    block.sync();  // Ensure all hashes complete before proceeding
}
#endif

// Memory bandwidth optimization macros
#define DARK_PREFETCH(addr) __builtin_prefetch(addr, 0, 3)
#define DARK_LIKELY(x) __builtin_expect(!!(x), 1)
#define DARK_UNLIKELY(x) __builtin_expect(!!(x), 0)

// Cache-optimized data structures
template<int N>
struct alignas(128) CacheOptimizedArray {
    uint8_t data[N];
    uint8_t padding[128 - (N % 128)];  // Align to cache line
};

// Advanced occupancy optimization
__device__ __forceinline__ void optimize_occupancy() {
    // Use fewer registers per thread to increase occupancy
    #pragma unroll 1  // Prevent aggressive loop unrolling
    // Prefer shared memory over registers for large data
}