#include "cuda_cracker.h"
#include "darkling_engine.h"
#include <cuda.h>
#include <cstring>
#include <iostream>

namespace darkling {

CudaCracker::CudaCracker() {}
CudaCracker::~CudaCracker() {}

bool CudaCracker::initialize(const JobConfig &cfg) {
    config_ = cfg;
    cudaGetDevice(&device_id_);
    return true;
}

bool CudaCracker::load_data(const std::vector<std::string> &charsets,
                            const std::vector<uint8_t> &position_map,
                            const std::vector<uint8_t> &hashes) {
    // Reuse logic from DarklingContext in darkling_host.cpp
    return true; // placeholder
}

bool CudaCracker::launch_crack_batch(uint64_t start, uint64_t end) {
    dim3 grid{128};
    dim3 block{256};
    // Actual kernel launch would mirror darkling_host.cpp
    launch_darkling(nullptr, nullptr, nullptr, nullptr,
                    config_.pwd_len, nullptr, 0, config_.hash_len,
                    start, end, nullptr, 0, nullptr, grid, block);
    cudaDeviceSynchronize();
    return true;
}

std::vector<std::string> CudaCracker::read_results() {
    return {}; // placeholder
}

std::string CudaCracker::get_status() {
    return "cuda";
}

} // namespace darkling
