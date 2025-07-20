#ifndef CUDA_CRACKER_H
#define CUDA_CRACKER_H

#include "gpu_backend.h"
#include <cuda_runtime.h>

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
    int device_id_ = 0;
    MaskJob job_{};
};

} // namespace darkling

#endif // CUDA_CRACKER_H
