#ifndef CUDA_CRACKER_H
#define CUDA_CRACKER_H

#include "gpu_backend.h"
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

namespace darkling {

class CudaCracker : public GpuCracker {
public:
    CudaCracker();
    ~CudaCracker() override;

    bool initialize() override;
    bool load_job(const MaskJob &job) override;
    bool run_batch() override;
    std::vector<CrackResult> read_results() override;
    GpuStatus get_status() override;

private:
    void upload_batch(int start_hash, int batch_size);
    void autotune(uint64_t start, uint64_t end);
    void allocate_buffers();

    int device_id_ = 0;
    MaskJob job_{};
    std::vector<std::vector<uint8_t>> charset_bytes_{};
    std::vector<std::vector<uint8_t>> charset_lens_{};
    std::vector<int> charset_sizes_{};
    std::vector<const uint8_t*> charset_byte_ptrs_{};
    std::vector<const uint8_t*> charset_len_ptrs_{};
    std::vector<uint8_t> pos_map_{};
    std::vector<uint8_t> all_hashes_{};
    uint8_t* d_all_hashes_ = nullptr;
    size_t d_all_hashes_size_ = 0;
    char* h_results_ = nullptr;
    char* d_results_ = nullptr;
    int* h_count_ = nullptr;
    int* d_count_ = nullptr;
    dim3 grid_{128};
    dim3 block_{256};
    bool tuned_ = false;
    std::vector<CrackResult> results_{};
    std::chrono::high_resolution_clock::time_point start_time_{};
    std::chrono::high_resolution_clock::time_point end_time_{};
};

} // namespace darkling

#endif // CUDA_CRACKER_H
