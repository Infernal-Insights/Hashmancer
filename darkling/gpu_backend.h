#ifndef GPU_BACKEND_H
#define GPU_BACKEND_H

#include <memory>
#include <vector>
#include <cstdint>
#include <string>

namespace darkling {

enum class GpuBackend {
    NVIDIA_CUDA,
    AMD_HIP,
    INTEL_OPENCL
};

struct JobConfig {
    int hash_len = 0;
    int pwd_len = 0;
};

struct GpuCracker {
    virtual ~GpuCracker() = default;
    virtual bool initialize(const JobConfig &config) = 0;
    virtual bool load_data(const std::vector<std::string> &charsets,
                           const std::vector<uint8_t> &position_map,
                           const std::vector<uint8_t> &hashes) = 0;
    virtual bool launch_crack_batch(uint64_t start, uint64_t end) = 0;
    virtual std::vector<std::string> read_results() = 0;
    virtual std::string get_status() = 0;
};

std::unique_ptr<GpuCracker> create_backend(GpuBackend type);

} // namespace darkling

#endif // GPU_BACKEND_H
