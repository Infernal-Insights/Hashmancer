#include "gpu_backend.h"
#include <iostream>
#include <cstring>

using namespace darkling;

static GpuBackend detect_backend() {
#ifdef __CUDACC__
    return GpuBackend::NVIDIA_CUDA;
#else
    const char* env = std::getenv("DARKLING_GPU_BACKEND");
    if (env && std::strcmp(env, "hip") == 0)
        return GpuBackend::AMD_HIP;
    if (env && std::strcmp(env, "opencl") == 0)
        return GpuBackend::INTEL_OPENCL;
#endif
    return GpuBackend::NVIDIA_CUDA;
}

int main(int argc, char** argv) {
    GpuBackend backend = detect_backend();
    auto cracker = create_backend(backend);
    JobConfig cfg{20, 4};
    cracker->initialize(cfg);
    cracker->load_data({}, {}, {});
    cracker->launch_crack_batch(0, 1000);
    auto res = cracker->read_results();
    for (auto& s : res) std::cout << s << "\n";
    std::cout << cracker->get_status() << std::endl;
    return 0;
}
