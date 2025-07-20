#ifndef HIP_CRACKER_H
#define HIP_CRACKER_H

#include "gpu_backend.h"

namespace darkling {

class HipCracker : public GpuCracker {
public:
    HipCracker();
    ~HipCracker() override;

    bool initialize() override;
    bool load_job(const MaskJob &job) override;
    bool run_batch() override;
    std::vector<CrackResult> read_results() override;
    GpuStatus get_status() override;
};

} // namespace darkling

#endif // HIP_CRACKER_H
