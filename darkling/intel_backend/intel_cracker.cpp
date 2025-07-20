#include "intel_cracker.h"
#include <iostream>

namespace darkling {

IntelCracker::IntelCracker() {}
IntelCracker::~IntelCracker() {}

bool IntelCracker::initialize() {
    return true;
}

bool IntelCracker::load_job(const MaskJob &job) {
    (void)job;
    return true;
}

bool IntelCracker::run_batch() {
    return true;
}

std::vector<CrackResult> IntelCracker::read_results() {
    return {};
}

GpuStatus IntelCracker::get_status() {
    return {};
}

} // namespace darkling
