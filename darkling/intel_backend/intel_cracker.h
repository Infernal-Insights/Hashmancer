#ifndef INTEL_CRACKER_H
#define INTEL_CRACKER_H

#include "gpu_backend.h"

namespace darkling {

class IntelCracker : public GpuCracker {
public:
    IntelCracker();
    ~IntelCracker() override;

    bool initialize() override;
    bool load_job(const MaskJob &job) override;
    bool run_batch() override;
    std::vector<CrackResult> read_results() override;
    GpuStatus get_status() override;
};

} // namespace darkling

#endif // INTEL_CRACKER_H
