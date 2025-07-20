#include "cuda_cracker.h"
#include "darkling_engine.h"
#include <cuda.h>
#include <cstring>
#include <iostream>

namespace darkling {

CudaCracker::CudaCracker() {}
CudaCracker::~CudaCracker() {}

bool CudaCracker::initialize() {
    cudaGetDevice(&device_id_);
    return true;
}

bool CudaCracker::load_job(const MaskJob &job) {
    job_ = job;
    // Reuse logic from DarklingContext in darkling_host.cpp
    return true; // placeholder
}

bool CudaCracker::run_batch() {
    dim3 grid{128};
    dim3 block{256};
    // Actual kernel launch would mirror darkling_host.cpp
    launch_darkling(nullptr, nullptr, nullptr, nullptr,
                    job_.mask_length, nullptr, 0, job_.hash_length,
                    job_.start_index, job_.end_index, nullptr, 0, nullptr, grid, block);
    cudaDeviceSynchronize();
    return true;
}

std::vector<CrackResult> CudaCracker::read_results() {
    return {}; // placeholder
}

GpuStatus CudaCracker::get_status() {
    return {};
}

} // namespace darkling
