#include "hip_cracker.h"
#include <iostream>

namespace darkling {

HipCracker::HipCracker() {}
HipCracker::~HipCracker() {}

bool HipCracker::initialize() {
    return true;
}

bool HipCracker::load_job(const MaskJob &job) {
    (void)job;
    return true;
}

bool HipCracker::run_batch() {
    return true;
}

std::vector<CrackResult> HipCracker::read_results() {
    return {};
}

GpuStatus HipCracker::get_status() {
    return {};
}

} // namespace darkling
