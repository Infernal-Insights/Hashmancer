#include "gpu_backend.h"
#include <iostream>
#include <cstring>
#include <dlfcn.h>

using namespace darkling;

static GpuBackend detect_backend(const char* override_backend = nullptr) {
#ifdef __CUDACC__
    (void)override_backend;
    return GpuBackend::NVIDIA_CUDA;
#else
    const char* env = override_backend ? override_backend : std::getenv("DARKLING_GPU_BACKEND");
    if (env && std::strcmp(env, "hip") == 0)
        return GpuBackend::AMD_HIP;
    if (env && std::strcmp(env, "opencl") == 0)
        return GpuBackend::INTEL_OPENCL;

    void* lib = dlopen("libcuda.so", RTLD_LAZY);
    if (lib) { dlclose(lib); return GpuBackend::NVIDIA_CUDA; }
    lib = dlopen("libamdhip64.so", RTLD_LAZY);
    if (!lib) lib = dlopen("libhip_hcc.so", RTLD_LAZY);
    if (lib) { dlclose(lib); return GpuBackend::AMD_HIP; }
    lib = dlopen("libOpenCL.so", RTLD_LAZY);
    if (lib) { dlclose(lib); return GpuBackend::INTEL_OPENCL; }
#endif
    return GpuBackend::NVIDIA_CUDA;
}

extern "C" GpuBackend test_detect_backend(const char* override_backend) {
    return detect_backend(override_backend);
}

#ifndef DARKLING_NO_MAIN
int main(int argc, char** argv) {
    const char* override = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            override = argv[i + 1];
            ++i;
        }
    }

    GpuBackend backend = detect_backend(override);
    std::cerr << "[darkling] using backend "
              << (backend == GpuBackend::NVIDIA_CUDA ? "CUDA" :
                  backend == GpuBackend::AMD_HIP ? "HIP" : "OpenCL")
              << "\n";
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
#endif
