#include "gpu_backend.h"
#include "cuda_backend/cuda_cracker.h"
#include "hip_backend/hip_cracker.h"
#include "intel_backend/intel_cracker.h"
#include <memory>

namespace darkling {

std::unique_ptr<GpuCracker> create_backend(GpuBackend type) {
    switch (type) {
        case GpuBackend::NVIDIA_CUDA:
            return std::make_unique<CudaCracker>();
        case GpuBackend::AMD_HIP:
            return std::make_unique<HipCracker>();
        case GpuBackend::INTEL_OPENCL:
        default:
            return std::make_unique<IntelCracker>();
    }
}

} // namespace darkling
