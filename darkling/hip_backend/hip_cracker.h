#ifndef HIP_CRACKER_H
#define HIP_CRACKER_H

#include "gpu_backend.h"

namespace darkling {

class HipCracker : public GpuCracker {
public:
    HipCracker();
    ~HipCracker() override;

    bool initialize(const JobConfig &config) override;
    bool load_data(const std::vector<std::string> &charsets,
                   const std::vector<uint8_t> &position_map,
                   const std::vector<uint8_t> &hashes) override;
    bool launch_crack_batch(uint64_t start, uint64_t end) override;
    std::vector<std::string> read_results() override;
    std::string get_status() override;
};

} // namespace darkling

#endif // HIP_CRACKER_H
