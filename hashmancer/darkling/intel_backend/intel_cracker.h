#ifndef INTEL_CRACKER_H
#define INTEL_CRACKER_H

#include "gpu_backend.h"
#include <vector>
#include <chrono>

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

private:
    MaskJob job_{};
    std::vector<CrackResult> results_{};
    std::chrono::high_resolution_clock::time_point start_time_{};
    std::chrono::high_resolution_clock::time_point end_time_{};
};

} // namespace darkling

#endif // INTEL_CRACKER_H
