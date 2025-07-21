#include "hip_cracker.h"
#include "darkling_engine.h"
#include <hip/hip_runtime.h>
#include <iostream>

namespace darkling {

HipCracker::HipCracker() {}
HipCracker::~HipCracker() {}

bool HipCracker::initialize() {
    return true;
}

bool HipCracker::load_job(const MaskJob &job) {
    job_ = job;
    return true;
}

bool HipCracker::run_batch() {
    dim3 grid{64};
    dim3 block{64};

    static uint8_t cs_bytes[MAX_CUSTOM_SETS][MAX_CHARSET_CHARS][MAX_UTF8_BYTES];
    static uint8_t cs_lens[MAX_CUSTOM_SETS][MAX_CHARSET_CHARS];
    static int cs_sizes[MAX_CUSTOM_SETS];

    for(int s=0;s<MAX_CUSTOM_SETS;s++){
        int len=job_.charset_lengths[s];
        cs_sizes[s]=len;
        for(int i=0;i<len;i++){
            cs_bytes[s][i][0]=job_.charsets[s][i];
            cs_lens[s][i]=1;
        }
    }

    const uint8_t* byte_ptrs[MAX_CUSTOM_SETS];
    const uint8_t* len_ptrs[MAX_CUSTOM_SETS];
    for(int i=0;i<MAX_CUSTOM_SETS;i++){
        byte_ptrs[i]=&cs_bytes[i][0][0];
        len_ptrs[i]=&cs_lens[i][0];
    }

    char* d_results=nullptr;
    int* d_count=nullptr;
    hipMalloc(&d_results, MAX_RESULT_BUFFER * MAX_PWD_BYTES);
    hipMalloc(&d_count, sizeof(int));
    hipMemset(d_count, 0, sizeof(int));

    launch_darkling_hip(byte_ptrs, len_ptrs, cs_sizes,
                        job_.mask_template, job_.mask_length,
                        reinterpret_cast<const uint8_t*>(job_.hashes),
                        job_.num_hashes, job_.hash_length,
                        job_.start_index, job_.end_index,
                        d_results, MAX_RESULT_BUFFER, d_count,
                        grid, block);
    hipDeviceSynchronize();

    hipFree(d_results);
    hipFree(d_count);
    return true;
}

std::vector<CrackResult> HipCracker::read_results() {
    return {};
}

GpuStatus HipCracker::get_status() {
    return {};
}

} // namespace darkling
