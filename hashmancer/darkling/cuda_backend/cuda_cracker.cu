#include "cuda_cracker.h"
#include "darkling_engine.h"
#include <cuda.h>
#include <cstring>
#include <iostream>
#include <algorithm>

#define CUDA_CHECK(x) \
    do { cudaError_t err = (x); if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl; return false; } } while (0)

namespace darkling {

CudaCracker::CudaCracker() {}
CudaCracker::~CudaCracker() {
    if (d_all_hashes_) {
        cudaFree(d_all_hashes_);
        d_all_hashes_ = nullptr;
    }
    if (h_results_) {
        cudaFreeHost(h_results_);
        h_results_ = nullptr;
        d_results_ = nullptr;
    }
    if (h_count_) {
        cudaFreeHost(h_count_);
        h_count_ = nullptr;
        d_count_ = nullptr;
    }
}

bool CudaCracker::initialize() {
    cudaGetDevice(&device_id_);
    return true;
}

bool CudaCracker::load_job(const MaskJob &job) {
    job_ = job;
    results_.clear();

    allocate_buffers();

    pos_map_.assign(job.mask_template, job.mask_template + job.mask_length);

    charset_bytes_.clear();
    charset_lens_.clear();
    charset_sizes_.clear();
    for (int s = 0; s < MAX_CHARSETS; ++s) {
        int len = job.charset_lengths[s];
        charset_sizes_.push_back(len);
        std::vector<uint8_t> bytes;
        std::vector<uint8_t> lens;
        for (int i = 0; i < len; ++i) {
            bytes.push_back(job.charsets[s][i]);
            for (int j = 1; j < MAX_UTF8_BYTES; ++j) bytes.push_back(0);
            lens.push_back(1);
        }
        charset_bytes_.push_back(std::move(bytes));
        charset_lens_.push_back(std::move(lens));
    }
    charset_byte_ptrs_.clear();
    charset_len_ptrs_.clear();
    for (size_t i = 0; i < charset_bytes_.size(); ++i) {
        charset_byte_ptrs_.push_back(charset_bytes_[i].data());
        charset_len_ptrs_.push_back(charset_lens_[i].data());
    }

    size_t hash_sz = static_cast<size_t>(job.num_hashes) * job.hash_length;
    all_hashes_.assign(hash_sz, 0);
    std::memcpy(all_hashes_.data(), job.hashes, hash_sz);
    if (d_all_hashes_size_ != hash_sz) {
        if (d_all_hashes_) cudaFree(d_all_hashes_);
        CUDA_CHECK(cudaMalloc(&d_all_hashes_, hash_sz));
        d_all_hashes_size_ = hash_sz;
    }
    CUDA_CHECK(cudaMemcpy(d_all_hashes_, all_hashes_.data(), hash_sz, cudaMemcpyHostToDevice));

    upload_batch(0, std::min<int>(job_.num_hashes, MAX_HASHES));
    tuned_ = false;
    return true;
}

bool CudaCracker::run_batch() {
    start_time_ = std::chrono::high_resolution_clock::now();
    if (!tuned_) {
        uint64_t sample_end = job_.start_index +
                              std::min<uint64_t>(1000, job_.end_index - job_.start_index);
        autotune(job_.start_index, sample_end);
    }

    results_.clear();
    size_t result_off = 0;
    int remaining = MAX_RESULT_BUFFER;

    if(grid_.x == 0 || block_.x == 0)
        return false;

    for (int off = 0; off < job_.num_hashes && remaining > 0; off += MAX_HASHES) {
        int batch = std::min(job_.num_hashes - off, MAX_HASHES);
        upload_batch(off, batch);
        CUDA_CHECK(cudaMemset(d_count_, 0, sizeof(int)));
        char* d_ptr = d_results_ + result_off * MAX_PWD_BYTES;
        launch_darkling_kernel(job_.start_index, job_.end_index, d_ptr,
                               remaining, d_count_, grid_, block_);
        CUDA_CHECK(cudaDeviceSynchronize());

        int found = *h_count_;
        char* h_ptr = h_results_ + result_off * MAX_PWD_BYTES;
        for (int i = 0; i < found && i < remaining; ++i) {
            const char* pwd = h_ptr + i * MAX_PWD_BYTES;
            CrackResult r{};
            r.candidate_index = 0;
            r.length = static_cast<uint8_t>(std::strlen(pwd));
            std::memcpy(r.password, pwd, r.length);
            std::memset(r.hash, 0, sizeof(r.hash));
            results_.push_back(r);
        }
        remaining = MAX_RESULT_BUFFER - static_cast<int>(results_.size());
        result_off = results_.size();
    }

    end_time_ = std::chrono::high_resolution_clock::now();
    return true;
}

void CudaCracker::allocate_buffers() {
    if (h_results_) return;
    size_t res_size = static_cast<size_t>(MAX_RESULT_BUFFER) * MAX_PWD_BYTES;
    CUDA_CHECK(cudaHostAlloc(&h_results_, res_size, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_results_, h_results_, 0));
    CUDA_CHECK(cudaHostAlloc(&h_count_, sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_count_, h_count_, 0));
}

void CudaCracker::upload_batch(int start_hash, int batch_size) {
    CUDA_CHECK(cudaMemcpyToSymbol(d_pwd_len, &job_.mask_length, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_hash_len, &job_.hash_length, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_hash_type, &job_.hash_type, sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_hashes, &batch_size, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_pos_charset, pos_map_.data(), job_.mask_length));

    for (int i = 0; i < MAX_CUSTOM_SETS; ++i) {
        int size = i < (int)charset_sizes_.size() ? charset_sizes_[i] : 0;
        CUDA_CHECK(cudaMemcpyToSymbol(d_charset_lens, &size, sizeof(int), i * sizeof(int)));
        if (size > 0) {
            CUDA_CHECK(cudaMemcpyToSymbol(d_charset_bytes, charset_byte_ptrs_[i],
                               size * MAX_UTF8_BYTES,
                               i * MAX_CHARSET_CHARS * MAX_UTF8_BYTES));
            CUDA_CHECK(cudaMemcpyToSymbol(d_charset_charlen, charset_len_ptrs_[i], size,
                               i * MAX_CHARSET_CHARS));
        }
    }

    const uint8_t* slice = d_all_hashes_ +
        static_cast<size_t>(start_hash) * job_.hash_length;
    CUDA_CHECK(cudaMemcpyToSymbol(d_hashes, slice,
                       static_cast<size_t>(batch_size) * job_.hash_length,
                       0, cudaMemcpyDeviceToDevice));
}

void CudaCracker::autotune(uint64_t start, uint64_t end) {
    const char* env = std::getenv("DARKLING_AUTOTUNE");
    if (env && std::strcmp(env, "0") == 0) {
        tuned_ = true;
        return;
    }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id_));
    int maxThreads = prop.maxThreadsPerBlock;
    int active = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, crack_kernel,
                                                  maxThreads, 0));
    grid_.x = active * prop.multiProcessorCount;
    block_.x = maxThreads;

    cudaEvent_t s,e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
    int batch = std::min(job_.num_hashes, MAX_HASHES);
    upload_batch(0, batch);
    CUDA_CHECK(cudaMemset(d_count_, 0, sizeof(int)));
    CUDA_CHECK(cudaEventRecord(s));
    launch_darkling_kernel(start, end, d_results_, MAX_RESULT_BUFFER,
                           d_count_, grid_, block_);
    CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    float rate = (float)(end - start) / (ms / 1000.0f);
    std::cerr << "[darkling] initial speed " << rate << " H/s\n";
    tuned_ = true;
}

std::vector<CrackResult> CudaCracker::read_results() {
    auto out = results_;
    results_.clear();
    return out;
}

GpuStatus CudaCracker::get_status() {
    GpuStatus s{};
    s.hashes_processed = job_.end_index - job_.start_index;
    s.batch_duration_ms =
        std::chrono::duration<float, std::milli>(end_time_ - start_time_).count();
    s.gpu_temp_c = 0.0f;
    s.overheat_flag = false;
    return s;
}

} // namespace darkling
