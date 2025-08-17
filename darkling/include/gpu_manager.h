#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <string>
#include <atomic>
#include <mutex>

// Advanced GPU Management for Multi-GPU Scaling
// Provides intelligent workload distribution and resource monitoring

struct GPUInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
    bool is_available;
    float utilization_percent;
    float memory_utilization_percent;
    float temperature_celsius;
    float power_usage_watts;
};

struct WorkloadDistribution {
    int device_id;
    uint64_t keyspace_start;
    uint64_t keyspace_end;
    uint32_t batch_size;
    float estimated_completion_time;
    std::atomic<uint64_t> progress;
    std::atomic<bool> completed;
    std::atomic<bool> failed;
};

class GPUManager {
public:
    GPUManager();
    ~GPUManager();
    
    // GPU Discovery and Information
    bool initialize();
    std::vector<GPUInfo> get_available_gpus();
    bool is_multi_gpu_available();
    int get_gpu_count();
    GPUInfo get_gpu_info(int device_id);
    
    // Workload Distribution
    std::vector<WorkloadDistribution> distribute_workload(
        uint64_t total_keyspace, 
        const std::vector<int>& target_devices,
        float estimated_keys_per_second_per_gpu = 1000000.0f
    );
    
    // Resource Management
    bool allocate_gpu_memory(int device_id, size_t size, void** ptr);
    bool free_gpu_memory(int device_id, void* ptr);
    bool set_active_device(int device_id);
    
    // Performance Monitoring
    bool update_gpu_stats();
    float get_total_hash_rate();
    float get_device_hash_rate(int device_id);
    size_t get_total_memory_usage();
    size_t get_device_memory_usage(int device_id);
    
    // Load Balancing
    int get_least_loaded_gpu();
    int get_most_powerful_gpu();
    bool balance_workload(std::vector<WorkloadDistribution>& workloads);
    
    // Thermal and Power Management
    bool is_thermal_throttling_needed();
    bool reduce_workload_if_overheating();
    float get_total_power_consumption();
    
    // Error Handling and Recovery
    bool handle_gpu_error(int device_id, cudaError_t error);
    bool recover_failed_device(int device_id);
    std::vector<int> get_healthy_devices();

private:
    std::vector<GPUInfo> gpu_info_;
    std::mutex gpu_mutex_;
    std::atomic<bool> initialized_;
    
    // Performance tracking
    std::vector<std::atomic<uint64_t>> device_hash_counts_;
    std::vector<std::chrono::steady_clock::time_point> last_update_times_;
    
    // Memory tracking
    std::vector<std::vector<void*>> allocated_memory_;
    std::vector<std::atomic<size_t>> memory_usage_;
    
    // Internal methods
    bool query_device_properties(int device_id, GPUInfo& info);
    bool update_device_utilization(int device_id);
    bool update_device_thermal_info(int device_id);
    float calculate_device_performance_score(const GPUInfo& info);
    uint32_t calculate_optimal_batch_size(const GPUInfo& info, size_t available_memory);
};

// Multi-GPU Kernel Launcher
class MultiGPULauncher {
public:
    MultiGPULauncher(GPUManager* gpu_manager);
    ~MultiGPULauncher();
    
    // Attack Coordination
    bool launch_dictionary_attack_multi_gpu(
        const std::vector<int>& device_ids,
        const uint8_t* wordlist_data,
        const uint32_t* word_offsets,
        uint32_t word_count,
        const uint8_t* target_hashes,
        uint32_t hash_count,
        uint8_t hash_type
    );
    
    bool launch_mask_attack_multi_gpu(
        const std::vector<int>& device_ids,
        const char* mask,
        uint64_t keyspace_start,
        uint64_t keyspace_end,
        const uint8_t* target_hashes,
        uint32_t hash_count,
        uint8_t hash_type
    );
    
    bool launch_rule_attack_multi_gpu(
        const std::vector<int>& device_ids,
        const uint8_t* wordlist_data,
        const uint32_t* word_offsets,
        uint32_t word_count,
        const void* rule_data,
        uint32_t rule_count,
        const uint8_t* target_hashes,
        uint32_t hash_count,
        uint8_t hash_type
    );
    
    // Progress Monitoring
    float get_overall_progress();
    std::vector<float> get_device_progress();
    uint64_t get_total_candidates_tested();
    uint32_t get_total_hashes_found();
    
    // Control
    bool pause_all_devices();
    bool resume_all_devices();
    bool stop_all_devices();
    bool is_running();

private:
    GPUManager* gpu_manager_;
    std::vector<cudaStream_t> device_streams_;
    std::vector<std::atomic<bool>> device_active_;
    std::vector<std::atomic<uint64_t>> device_progress_;
    std::vector<std::atomic<uint32_t>> device_found_count_;
    std::atomic<bool> running_;
    std::mutex launcher_mutex_;
    
    bool initialize_device_streams();
    bool cleanup_device_streams();
    bool synchronize_all_devices();
};

// GPU Performance Profiler
class GPUProfiler {
public:
    struct PerformanceMetrics {
        float hash_rate_mhs;           // Million hashes per second
        float memory_bandwidth_gbps;   // Memory bandwidth utilization
        float gpu_utilization_percent; // GPU core utilization
        float memory_utilization_percent; // VRAM utilization
        float efficiency_score;        // Overall efficiency (0-100)
        float power_efficiency_mhs_per_watt; // Performance per watt
        std::chrono::milliseconds kernel_execution_time;
        std::chrono::milliseconds memory_transfer_time;
    };
    
    GPUProfiler();
    ~GPUProfiler();
    
    bool start_profiling(int device_id);
    bool stop_profiling(int device_id);
    PerformanceMetrics get_metrics(int device_id);
    
    // Benchmarking
    PerformanceMetrics benchmark_device(int device_id, const std::string& algorithm);
    std::vector<PerformanceMetrics> benchmark_all_devices(const std::string& algorithm);
    
    // Optimization Recommendations
    struct OptimizationSuggestion {
        std::string category;      // "memory", "compute", "thermal"
        std::string suggestion;    // Human-readable suggestion
        float potential_improvement; // Estimated % improvement
        int priority;             // 1-10 priority
    };
    
    std::vector<OptimizationSuggestion> analyze_performance(int device_id);

private:
    std::vector<cudaEvent_t> start_events_;
    std::vector<cudaEvent_t> stop_events_;
    std::vector<PerformanceMetrics> device_metrics_;
    std::vector<bool> profiling_active_;
};

// Utility Functions
namespace gpu_utils {
    // Convert CUDA error to human-readable string
    std::string cuda_error_string(cudaError_t error);
    
    // Get optimal grid/block dimensions for device
    dim3 calculate_optimal_grid_size(int device_id, uint32_t total_threads);
    dim3 calculate_optimal_block_size(int device_id);
    
    // Memory management helpers
    size_t get_available_memory(int device_id);
    size_t get_optimal_batch_size(int device_id, size_t per_item_memory);
    
    // Performance helpers
    float calculate_theoretical_peak_performance(int device_id, const std::string& algorithm);
    float calculate_memory_bandwidth_utilization(int device_id, size_t bytes_transferred, float time_ms);
    
    // Thermal helpers
    bool is_device_overheating(int device_id, float threshold_celsius = 85.0f);
    bool should_throttle_device(int device_id);
}