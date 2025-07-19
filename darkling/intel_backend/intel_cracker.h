#ifndef INTEL_CRACKER_H
#define INTEL_CRACKER_H

#include "gpu_backend.h"

namespace darkling {

class IntelCracker : public GpuCracker {
public:
    IntelCracker();
    ~IntelCracker() override;

    bool initialize(const JobConfig &config) override;
    bool load_data(const std::vector<std::string> &charsets,
                   const std::vector<uint8_t> &position_map,
                   const std::vector<uint8_t> &hashes) override;
    bool launch_crack_batch(uint64_t start, uint64_t end) override;
    std::vector<std::string> read_results() override;
    std::string get_status() override;
};

} // namespace darkling

#endif // INTEL_CRACKER_H
