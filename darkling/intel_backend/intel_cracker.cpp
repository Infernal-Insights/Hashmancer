#include "intel_cracker.h"
#include <iostream>

namespace darkling {

IntelCracker::IntelCracker() {}
IntelCracker::~IntelCracker() {}

bool IntelCracker::initialize(const JobConfig &config) {
    (void)config;
    return true;
}

bool IntelCracker::load_data(const std::vector<std::string> &charsets,
                             const std::vector<uint8_t> &position_map,
                             const std::vector<uint8_t> &hashes) {
    (void)charsets; (void)position_map; (void)hashes;
    return true;
}

bool IntelCracker::launch_crack_batch(uint64_t start, uint64_t end) {
    (void)start; (void)end;
    return true;
}

std::vector<std::string> IntelCracker::read_results() {
    return {};
}

std::string IntelCracker::get_status() {
    return "intel";
}

} // namespace darkling
