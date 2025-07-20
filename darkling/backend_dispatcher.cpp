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
    cracker->initialize();
    MaskJob job{};
    job.start_index = 0;
    job.end_index = 1000;
    job.mask_length = 0;
    job.hash_length = 20;
    job.num_hashes = 0;
    cracker->load_job(job);
    cracker->run_batch();
    auto res = cracker->read_results();
    for (auto& r : res) {
        std::cout << r.candidate_index << "\n";
    }
    auto status = cracker->get_status();
    std::cout << status.hashes_processed << std::endl;
    return 0;
}
