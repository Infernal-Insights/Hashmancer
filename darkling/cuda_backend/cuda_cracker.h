#ifndef CUDA_CRACKER_H
#define CUDA_CRACKER_H

#include "gpu_backend.h"
#include <cuda_runtime.h>

namespace darkling {

class CudaCracker : public GpuCracker {
public:
    CudaCracker();
    ~CudaCracker() override;

    bool initialize(const JobConfig &config) override;
    bool load_data(const std::vector<std::string> &charsets,
                   const std::vector<uint8_t> &position_map,
                   const std::vector<uint8_t> &hashes) override;
    bool launch_crack_batch(uint64_t start, uint64_t end) override;
    std::vector<std::string> read_results() override;
    std::string get_status() override;

private:
    int device_id_ = 0;
    JobConfig config_{};
};

} // namespace darkling

#endif // CUDA_CRACKER_H
