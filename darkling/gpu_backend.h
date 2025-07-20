#ifndef GPU_BACKEND_H
#define GPU_BACKEND_H

#include <memory>
#include <vector>
#include <cstdint>
#include <string>
#include "gpu_shared_types.h"

namespace darkling {

enum class GpuBackend {
    NVIDIA_CUDA,
    AMD_HIP,
    INTEL_OPENCL
};

struct GpuCracker {
    virtual ~GpuCracker() = default;
    virtual bool initialize() = 0;
    virtual bool load_job(const MaskJob &job) = 0;
    virtual bool run_batch() = 0;
    virtual std::vector<CrackResult> read_results() = 0;
    virtual GpuStatus get_status() = 0;
};

std::unique_ptr<GpuCracker> create_backend(GpuBackend type);

} // namespace darkling

#endif // GPU_BACKEND_H
