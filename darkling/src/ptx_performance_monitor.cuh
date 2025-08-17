#pragma once
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Performance monitoring for PTX-optimized kernels
class PTXPerformanceMonitor {
private:
    cudaEvent_t start_event, stop_event;
    float kernel_time_ms;
    
public:
    PTXPerformanceMonitor() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }
    
    ~PTXPerformanceMonitor() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start_timing() {
        cudaEventRecord(start_event);
    }
    
    void stop_timing() {
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);
    }
    
    float get_kernel_time_ms() const {
        return kernel_time_ms;
    }
    
    // Calculate theoretical peak performance
    static void analyze_performance(int device_id, float achieved_time_ms, uint64_t operations) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        
        // Calculate theoretical metrics
        float memory_bandwidth_gbps = prop.memoryBusWidth * prop.memoryClockRate * 2.0f / 8.0f / 1e6f;
        float peak_gflops = prop.multiProcessorCount * prop.warpSize * prop.clockRate * 2.0f / 1e6f;
        
        // Calculate achieved performance
        float achieved_gops = operations / (achieved_time_ms * 1e6f);
        float memory_efficiency = (achieved_gops * sizeof(uint32_t)) / memory_bandwidth_gbps * 100.0f;
        
        printf("PTX Performance Analysis:\n");
        printf("  Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
        printf("  Peak Memory BW: %.1f GB/s\n", memory_bandwidth_gbps);
        printf("  Peak Compute: %.1f GFLOPS\n", peak_gflops);
        printf("  Achieved: %.1f GOP/s\n", achieved_gops);
        printf("  Memory Efficiency: %.1f%%\n", memory_efficiency);
        printf("  Kernel Time: %.3f ms\n", achieved_time_ms);
    }
};

// Compile-time PTX optimization checks
#define CHECK_PTX_OPTIMIZATION() \
    static_assert(__CUDA_ARCH__ >= 750, "PTX optimizations require SM 7.5+"); \
    static_assert(sizeof(float4) == 16, "Vectorization alignment check failed");

// Runtime PTX capability detection
__device__ __forceinline__ bool ptx_supports_vectorized_ops() {
    #if __CUDA_ARCH__ >= 750
    return true;
    #else
    return false;
    #endif
}

__device__ __forceinline__ bool ptx_supports_cooperative_groups() {
    #if __CUDA_ARCH__ >= 600
    return true;
    #else
    return false;
    #endif
}

// PTX rule performance hints
enum PTXRuleComplexity {
    PTX_SIMPLE_COPY = 1,     // Vectorized memory copy
    PTX_CHAR_TRANSFORM = 2,  // Character-level transformations
    PTX_LOOKUP_TABLE = 3,    // Table-based substitutions
    PTX_ARITHMETIC = 4,      // Mathematical operations
    PTX_COMPLEX_PATTERN = 5  // Multi-step transformations
};

// Estimate optimal batch size based on rule complexity
__host__ int estimate_optimal_batch_size(PTXRuleComplexity complexity, int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    
    int base_batch_size = prop.multiProcessorCount * prop.warpSize * 4;
    
    switch (complexity) {
        case PTX_SIMPLE_COPY:
            return base_batch_size * 4;  // Memory-bound, larger batches
        case PTX_CHAR_TRANSFORM:
            return base_batch_size * 3;
        case PTX_LOOKUP_TABLE:
            return base_batch_size * 2;
        case PTX_ARITHMETIC:
            return base_batch_size;
        case PTX_COMPLEX_PATTERN:
            return base_batch_size / 2;  // Compute-bound, smaller batches
        default:
            return base_batch_size;
    }
}

// PTX instruction throughput optimization hints
#define PTX_OPTIMIZE_THROUGHPUT() \
    __pragma("unroll") \
    __pragma("GCC optimize (\"O3,fast-math,unroll-loops\")")

// Memory access pattern optimization
template<typename T>
__device__ __forceinline__ void ptx_prefetch_l1(const T* addr) {
    #if __CUDA_ARCH__ >= 750
    __builtin_prefetch(addr, 0, 1);  // Prefetch to L1 cache
    #endif
}

template<typename T>
__device__ __forceinline__ void ptx_prefetch_l2(const T* addr) {
    #if __CUDA_ARCH__ >= 750
    __builtin_prefetch(addr, 0, 2);  // Prefetch to L2 cache
    #endif
}

// PTX register pressure optimization
#define PTX_LIMIT_REGISTERS(n) \
    __launch_bounds__(1024, n)

// Warp-level performance optimization
__device__ __forceinline__ uint32_t ptx_warp_ballot_performance(int predicate) {
    #if __CUDA_ARCH__ >= 700
    return __ballot_sync(0xFFFFFFFF, predicate);
    #else
    return __ballot(predicate);
    #endif
}

// Memory bandwidth utilization tracking
struct PTXMemoryStats {
    uint64_t bytes_read;
    uint64_t bytes_written;
    uint64_t cache_hits;
    uint64_t cache_misses;
};

__device__ void update_memory_stats(PTXMemoryStats* stats, uint64_t read_bytes, uint64_t write_bytes) {
    atomicAdd(&stats->bytes_read, read_bytes);
    atomicAdd(&stats->bytes_written, write_bytes);
}

// PTX rule auto-tuning framework
struct PTXAutoTuneParams {
    int block_size;
    int grid_size;
    int shared_mem_size;
    int registers_per_thread;
    bool use_vectorization;
    bool use_constant_memory;
};

__host__ PTXAutoTuneParams auto_tune_ptx_kernel(int device_id, PTXRuleComplexity complexity) {
    PTXAutoTuneParams params;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    
    // Auto-tune based on GPU architecture and rule complexity
    if (prop.major >= 8) {  // Ampere+
        params.block_size = (complexity <= PTX_LOOKUP_TABLE) ? 512 : 256;
        params.grid_size = prop.multiProcessorCount * 4;
        params.shared_mem_size = 49152;  // 48KB shared memory
        params.registers_per_thread = 64;
        params.use_vectorization = true;
        params.use_constant_memory = true;
    } else if (prop.major == 7) {  // Turing
        params.block_size = (complexity <= PTX_CHAR_TRANSFORM) ? 256 : 128;
        params.grid_size = prop.multiProcessorCount * 3;
        params.shared_mem_size = 32768;   // 32KB shared memory
        params.registers_per_thread = 48;
        params.use_vectorization = true;
        params.use_constant_memory = true;
    } else {  // Older architectures
        params.block_size = 128;
        params.grid_size = prop.multiProcessorCount * 2;
        params.shared_mem_size = 16384;   // 16KB shared memory
        params.registers_per_thread = 32;
        params.use_vectorization = false;
        params.use_constant_memory = false;
    }
    
    return params;
}