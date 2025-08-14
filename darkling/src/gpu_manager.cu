#include "gpu_manager.h"
#include <cuda_runtime.h>
#include <nvml.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>

// GPU Manager Implementation

GPUManager::GPUManager() : initialized_(false) {}

GPUManager::~GPUManager() {
    if (initialized_) {
        // Cleanup allocated memory
        for (int i = 0; i < gpu_info_.size(); ++i) {
            for (void* ptr : allocated_memory_[i]) {
                cudaSetDevice(i);
                cudaFree(ptr);
            }
        }
    }
}

bool GPUManager::initialize() {
    if (initialized_) return true;
    
    std::lock_guard<std::mutex> lock(gpu_mutex_);
    
    // Initialize NVML for advanced GPU monitoring
    nvmlReturn_t nvml_result = nvmlInit();
    if (nvml_result != NVML_SUCCESS) {
        std::cerr << "Warning: NVML initialization failed. Advanced monitoring disabled." << std::endl;
    }
    
    // Get GPU count
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        std::cerr << "Failed to get CUDA device count: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    if (device_count == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return false;
    }
    
    // Initialize data structures
    gpu_info_.resize(device_count);
    device_hash_counts_.resize(device_count);
    last_update_times_.resize(device_count);
    allocated_memory_.resize(device_count);
    memory_usage_.resize(device_count);
    
    // Query each device
    for (int i = 0; i < device_count; ++i) {
        if (!query_device_properties(i, gpu_info_[i])) {
            std::cerr << "Failed to query properties for device " << i << std::endl;
            continue;
        }
        
        device_hash_counts_[i] = 0;
        memory_usage_[i] = 0;
        last_update_times_[i] = std::chrono::steady_clock::now();
        
        std::cout << "GPU " << i << ": " << gpu_info_[i].name 
                  << " (" << gpu_info_[i].total_memory / (1024*1024) << " MB)" << std::endl;
    }
    
    initialized_ = true;
    return true;
}

bool GPUManager::query_device_properties(int device_id, GPUInfo& info) {
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, device_id);
    if (error != cudaSuccess) {
        return false;
    }
    
    info.device_id = device_id;
    info.name = std::string(prop.name);
    info.total_memory = prop.totalGlobalMem;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
    
    // Get current memory info
    size_t free_mem, total_mem;
    cudaSetDevice(device_id);
    error = cudaMemGetInfo(&free_mem, &total_mem);
    if (error == cudaSuccess) {
        info.free_memory = free_mem;
        info.is_available = free_mem > (100 * 1024 * 1024); // At least 100MB free
    } else {
        info.free_memory = 0;
        info.is_available = false;
    }
    
    // Update utilization and thermal info
    update_device_utilization(device_id);
    update_device_thermal_info(device_id);
    
    return true;
}

bool GPUManager::update_device_utilization(int device_id) {
    // Try to get utilization via NVML first
    nvmlDevice_t device;
    if (nvmlDeviceGetHandleByIndex(device_id, &device) == NVML_SUCCESS) {
        nvmlUtilization_t utilization;
        if (nvmlDeviceGetUtilizationRates(device, &utilization) == NVML_SUCCESS) {
            gpu_info_[device_id].utilization_percent = utilization.gpu;
            gpu_info_[device_id].memory_utilization_percent = utilization.memory;
            return true;
        }
    }
    
    // Fallback to estimating based on our hash rate measurements
    auto now = std::chrono::steady_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_update_times_[device_id]).count();
    
    if (time_diff > 1000) { // Update every second
        uint64_t hash_count = device_hash_counts_[device_id].exchange(0);
        float hash_rate = (hash_count * 1000.0f) / time_diff; // Hashes per second
        
        // Estimate utilization based on theoretical peak performance
        float theoretical_peak = calculate_theoretical_peak_performance(device_id, "md5");
        gpu_info_[device_id].utilization_percent = std::min(100.0f, (hash_rate / theoretical_peak) * 100.0f);
        
        last_update_times_[device_id] = now;
    }
    
    return true;
}

bool GPUManager::update_device_thermal_info(int device_id) {
    nvmlDevice_t device;
    if (nvmlDeviceGetHandleByIndex(device_id, &device) == NVML_SUCCESS) {
        unsigned int temp;
        if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
            gpu_info_[device_id].temperature_celsius = temp;
        }
        
        unsigned int power;
        if (nvmlDeviceGetPowerUsage(device, &power) == NVML_SUCCESS) {
            gpu_info_[device_id].power_usage_watts = power / 1000.0f; // Convert mW to W
        }
        
        return true;
    }
    
    return false;
}

std::vector<GPUInfo> GPUManager::get_available_gpus() {
    if (!initialized_) initialize();
    
    std::vector<GPUInfo> available;
    for (const auto& info : gpu_info_) {
        if (info.is_available) {
            available.push_back(info);
        }
    }
    return available;
}

bool GPUManager::is_multi_gpu_available() {
    return get_available_gpus().size() > 1;
}

int GPUManager::get_gpu_count() {
    return gpu_info_.size();
}

GPUInfo GPUManager::get_gpu_info(int device_id) {
    if (device_id >= 0 && device_id < gpu_info_.size()) {
        update_device_utilization(device_id);
        update_device_thermal_info(device_id);
        return gpu_info_[device_id];
    }
    return GPUInfo{};
}

std::vector<WorkloadDistribution> GPUManager::distribute_workload(
    uint64_t total_keyspace, 
    const std::vector<int>& target_devices,
    float estimated_keys_per_second_per_gpu) {
    
    std::vector<WorkloadDistribution> distributions;
    
    if (target_devices.empty() || total_keyspace == 0) {
        return distributions;
    }
    
    // Calculate performance scores for each device
    std::vector<float> performance_scores;
    float total_performance = 0.0f;
    
    for (int device_id : target_devices) {
        if (device_id < gpu_info_.size()) {
            float score = calculate_device_performance_score(gpu_info_[device_id]);
            performance_scores.push_back(score);
            total_performance += score;
        } else {
            performance_scores.push_back(0.0f);
        }
    }
    
    // Distribute keyspace proportionally to performance
    uint64_t allocated_keyspace = 0;
    
    for (size_t i = 0; i < target_devices.size(); ++i) {
        WorkloadDistribution dist;
        dist.device_id = target_devices[i];
        dist.keyspace_start = allocated_keyspace;
        
        if (i == target_devices.size() - 1) {
            // Last device gets remaining keyspace
            dist.keyspace_end = total_keyspace;
        } else {
            // Proportional allocation
            uint64_t device_keyspace = static_cast<uint64_t>(
                (performance_scores[i] / total_performance) * total_keyspace
            );
            dist.keyspace_end = allocated_keyspace + device_keyspace;
        }
        
        // Calculate optimal batch size based on device memory
        const GPUInfo& info = gpu_info_[dist.device_id];
        dist.batch_size = calculate_optimal_batch_size(info, info.free_memory);
        
        // Estimate completion time
        uint64_t device_work = dist.keyspace_end - dist.keyspace_start;
        float device_rate = estimated_keys_per_second_per_gpu * performance_scores[i];
        dist.estimated_completion_time = device_work / device_rate;
        
        dist.progress = 0;
        dist.completed = false;
        dist.failed = false;
        
        distributions.push_back(dist);
        allocated_keyspace = dist.keyspace_end;
    }
    
    return distributions;
}

float GPUManager::calculate_device_performance_score(const GPUInfo& info) {
    // Base score from compute capability and SM count
    float base_score = info.multiprocessor_count * 
                      (info.compute_capability_major * 10 + info.compute_capability_minor);
    
    // Memory bandwidth factor (approximate)
    float memory_factor = std::sqrt(info.total_memory / (1024.0f * 1024.0f * 1024.0f)); // GB
    
    // Availability factor (reduce score if device is busy)
    float availability_factor = (100.0f - info.utilization_percent) / 100.0f;
    
    // Thermal factor (reduce score if overheating)
    float thermal_factor = 1.0f;
    if (info.temperature_celsius > 80.0f) {
        thermal_factor = std::max(0.5f, (90.0f - info.temperature_celsius) / 10.0f);
    }
    
    return base_score * memory_factor * availability_factor * thermal_factor;
}

uint32_t GPUManager::calculate_optimal_batch_size(const GPUInfo& info, size_t available_memory) {
    // Reserve 20% of memory for other operations
    size_t usable_memory = static_cast<size_t>(available_memory * 0.8);
    
    // Estimate memory per hash candidate (varies by algorithm)
    size_t memory_per_candidate = 64; // Conservative estimate
    
    uint32_t max_candidates = static_cast<uint32_t>(usable_memory / memory_per_candidate);
    
    // Align to block size for optimal GPU utilization
    uint32_t block_size = 256; // Common block size
    uint32_t blocks_per_sm = 4; // Conservative estimate
    
    uint32_t optimal_candidates = info.multiprocessor_count * blocks_per_sm * block_size;
    
    return std::min(max_candidates, optimal_candidates);
}

int GPUManager::get_least_loaded_gpu() {
    update_gpu_stats();
    
    int best_gpu = -1;
    float lowest_utilization = 100.0f;
    
    for (const auto& info : gpu_info_) {
        if (info.is_available && info.utilization_percent < lowest_utilization) {
            lowest_utilization = info.utilization_percent;
            best_gpu = info.device_id;
        }
    }
    
    return best_gpu;
}

int GPUManager::get_most_powerful_gpu() {
    float highest_score = 0.0f;
    int best_gpu = -1;
    
    for (const auto& info : gpu_info_) {
        if (info.is_available) {
            float score = calculate_device_performance_score(info);
            if (score > highest_score) {
                highest_score = score;
                best_gpu = info.device_id;
            }
        }
    }
    
    return best_gpu;
}

bool GPUManager::update_gpu_stats() {
    for (int i = 0; i < gpu_info_.size(); ++i) {
        update_device_utilization(i);
        update_device_thermal_info(i);
        
        // Update memory info
        size_t free_mem, total_mem;
        cudaSetDevice(i);
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
            gpu_info_[i].free_memory = free_mem;
            gpu_info_[i].memory_utilization_percent = 
                ((float)(total_mem - free_mem) / total_mem) * 100.0f;
        }
    }
    return true;
}

bool GPUManager::is_thermal_throttling_needed() {
    for (const auto& info : gpu_info_) {
        if (info.is_available && info.temperature_celsius > 85.0f) {
            return true;
        }
    }
    return false;
}

bool GPUManager::reduce_workload_if_overheating() {
    bool throttled = false;
    
    for (auto& info : gpu_info_) {
        if (info.is_available && info.temperature_celsius > 85.0f) {
            // Reduce workload by marking as less available
            info.utilization_percent = std::min(info.utilization_percent * 1.2f, 100.0f);
            throttled = true;
        }
    }
    
    return throttled;
}

// Utility function implementations
namespace gpu_utils {
    std::string cuda_error_string(cudaError_t error) {
        return std::string(cudaGetErrorString(error));
    }
    
    dim3 calculate_optimal_grid_size(int device_id, uint32_t total_threads) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        
        int block_size = 256; // Common optimal block size
        int grid_size = (total_threads + block_size - 1) / block_size;
        
        // Limit grid size to device capabilities
        grid_size = std::min(grid_size, prop.maxGridSize[0]);
        
        return dim3(grid_size, 1, 1);
    }
    
    dim3 calculate_optimal_block_size(int device_id) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        
        // Use a block size that's a multiple of the warp size (32)
        // and fits well with the device's capabilities
        int block_size = std::min(256, prop.maxThreadsPerBlock);
        block_size = (block_size / 32) * 32; // Round down to warp boundary
        
        return dim3(block_size, 1, 1);
    }
    
    size_t get_available_memory(int device_id) {
        cudaSetDevice(device_id);
        size_t free_mem, total_mem;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
            return free_mem;
        }
        return 0;
    }
    
    float calculate_theoretical_peak_performance(int device_id, const std::string& algorithm) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        
        // Very rough estimates based on device specs
        float base_performance = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
        
        if (algorithm == "md5") {
            return base_performance * 1000000.0f; // ~1M hashes per core per second
        } else if (algorithm == "sha1") {
            return base_performance * 800000.0f;
        } else if (algorithm == "ntlm") {
            return base_performance * 1200000.0f;
        }
        
        return base_performance * 500000.0f; // Conservative default
    }
    
    bool is_device_overheating(int device_id, float threshold_celsius) {
        nvmlDevice_t device;
        if (nvmlDeviceGetHandleByIndex(device_id, &device) == NVML_SUCCESS) {
            unsigned int temp;
            if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
                return temp > threshold_celsius;
            }
        }
        return false;
    }
}