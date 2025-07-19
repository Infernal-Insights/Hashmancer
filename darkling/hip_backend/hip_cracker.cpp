#include "hip_cracker.h"
#include <iostream>

namespace darkling {

HipCracker::HipCracker() {}
HipCracker::~HipCracker() {}

bool HipCracker::initialize(const JobConfig &config) {
    (void)config;
    return true;
}

bool HipCracker::load_data(const std::vector<std::string> &charsets,
                           const std::vector<uint8_t> &position_map,
                           const std::vector<uint8_t> &hashes) {
    (void)charsets; (void)position_map; (void)hashes;
    return true;
}

bool HipCracker::launch_crack_batch(uint64_t start, uint64_t end) {
    (void)start; (void)end;
    return true;
}

std::vector<std::string> HipCracker::read_results() {
    return {};
}

std::string HipCracker::get_status() {
    return "hip";
}

} // namespace darkling
