#include "hip_cracker.h"
#include "darkling_engine.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>

namespace darkling {

HipCracker::HipCracker() {}
HipCracker::~HipCracker() {}

bool HipCracker::initialize() {
    return true;
}

bool HipCracker::load_job(const MaskJob &job) {
    job_ = job;

    // Convert and upload charsets and hash buffers to constant memory
    static uint8_t cs_bytes[MAX_CUSTOM_SETS][MAX_CHARSET_CHARS][MAX_UTF8_BYTES];
    static uint8_t cs_lens[MAX_CUSTOM_SETS][MAX_CHARSET_CHARS];
    static int cs_sizes[MAX_CUSTOM_SETS];

    for(int s = 0; s < MAX_CUSTOM_SETS; ++s) {
        int len = job_.charset_lengths[s];
        cs_sizes[s] = len;
        for(int i = 0; i < len; ++i) {
            // job_ charset bytes are ASCII - store as UTF8 with length 1
            cs_bytes[s][i][0] = job_.charsets[s][i];
            for(int j=1;j<MAX_UTF8_BYTES;++j) cs_bytes[s][i][j] = 0;
            cs_lens[s][i] = 1;
        }
    }

    hipMemcpyToSymbol(HIP_SYMBOL(d_pwd_len), &job_.mask_length, sizeof(int));
    hipMemcpyToSymbol(HIP_SYMBOL(d_hash_len), &job_.hash_length, sizeof(int));
    hipMemcpyToSymbol(HIP_SYMBOL(d_hash_type), &job_.hash_type, sizeof(uint8_t));
    hipMemcpyToSymbol(HIP_SYMBOL(d_num_hashes), &job_.num_hashes, sizeof(int));
    hipMemcpyToSymbol(HIP_SYMBOL(d_pos_charset), job_.mask_template, job_.mask_length);

    for(int s=0; s<MAX_CUSTOM_SETS; ++s) {
        hipMemcpyToSymbol(HIP_SYMBOL(d_charset_lens), &cs_sizes[s], sizeof(int), s*sizeof(int));
        if(cs_sizes[s] > 0) {
            hipMemcpyToSymbol(HIP_SYMBOL(d_charset_bytes), cs_bytes[s], cs_sizes[s]*MAX_UTF8_BYTES,
                               s*MAX_CHARSET_CHARS*MAX_UTF8_BYTES);
            hipMemcpyToSymbol(HIP_SYMBOL(d_charset_charlen), cs_lens[s], cs_sizes[s],
                               s*MAX_CHARSET_CHARS);
        }
    }
    hipMemcpyToSymbol(HIP_SYMBOL(d_hashes), job_.hashes, static_cast<size_t>(job_.num_hashes) * job_.hash_length);

    return true;
}

bool HipCracker::run_batch() {
    dim3 grid{64};
    dim3 block{64};

    size_t res_size = static_cast<size_t>(MAX_RESULT_BUFFER) * MAX_PWD_BYTES;
    char* d_results = nullptr;
    int* d_count = nullptr;
    hipMalloc(&d_results, res_size);
    hipMalloc(&d_count, sizeof(int));
    hipMemset(d_count, 0, sizeof(int));

    uint64_t total = job_.end_index - job_.start_index;
    hipLaunchKernelGGL(crack_kernel, grid, block, 0, 0,
                       job_.start_index, total, d_results,
                       MAX_RESULT_BUFFER, d_count);
    hipDeviceSynchronize();

    int h_count = 0;
    hipMemcpy(&h_count, d_count, sizeof(int), hipMemcpyDeviceToHost);
    h_count = std::min(h_count, MAX_RESULT_BUFFER);
    std::vector<char> buffer(static_cast<size_t>(h_count) * MAX_PWD_BYTES);
    if(h_count > 0)
        hipMemcpy(buffer.data(), d_results,
                  static_cast<size_t>(h_count) * MAX_PWD_BYTES,
                  hipMemcpyDeviceToHost);

    results_.clear();
    for(int i=0;i<h_count;i++) {
        const char* pwd = buffer.data() + static_cast<size_t>(i)*MAX_PWD_BYTES;
        CrackResult r{};
        r.candidate_index = 0;
        r.length = static_cast<uint8_t>(std::strlen(pwd));
        std::memcpy(r.password, pwd, r.length);
        std::memset(r.hash, 0, sizeof(r.hash));
        results_.push_back(r);
    }

    hipFree(d_results);
    hipFree(d_count);
    return true;
}

std::vector<CrackResult> HipCracker::read_results() {
    auto out = results_;
    results_.clear();
    return out;
}

GpuStatus HipCracker::get_status() {
    GpuStatus s{};
    s.hashes_processed = job_.end_index - job_.start_index;
    s.gpu_temp_c = 0.0f;
    s.batch_duration_ms = 0.0f;
    s.overheat_flag = false;
    return s;
}

} // namespace darkling
