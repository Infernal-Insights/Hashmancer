#include "darkling_engine.h"
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>

struct DarklingContext {
    int pwd_len = 0;
    int hash_len = 0;
    int num_hashes = 0;
    int max_results = 0;

    std::vector<const char*> charset_ptrs;
    std::vector<int> charset_lens;
    std::vector<uint8_t> hashes;

    char* h_results = nullptr;
    char* d_results = nullptr;
    int* h_count = nullptr;
    int* d_count = nullptr;

    dim3 grid{128};
    dim3 block{256};

    void allocate_buffers(int max_res) {
        max_results = max_res;
        size_t res_size = static_cast<size_t>(max_results) * MAX_MASK_LEN;
        cudaHostAlloc(&h_results, res_size, cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_results, h_results, 0);
        cudaHostAlloc(&h_count, sizeof(int), cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_count, h_count, 0);
    }

    void preload(const std::vector<std::string>& charsets,
                 const std::vector<uint8_t>& hsh,
                 int plen, int hlen) {
        pwd_len = plen;
        hash_len = hlen;
        num_hashes = hsh.size() / hlen;
        hashes = hsh;

        charset_ptrs.clear();
        charset_lens.clear();
        for (const auto& cs : charsets) {
            charset_ptrs.push_back(cs.c_str());
            charset_lens.push_back(static_cast<int>(cs.size()));
        }

        cudaMemcpyToSymbol(d_pwd_len, &pwd_len, sizeof(int));
        cudaMemcpyToSymbol(d_hash_len, &hash_len, sizeof(int));
        cudaMemcpyToSymbol(d_num_hashes, &num_hashes, sizeof(int));
        for (int i = 0; i < pwd_len; ++i) {
            cudaMemcpyToSymbol(d_charsets[i], charset_ptrs[i], charset_lens[i], 0);
            cudaMemcpyToSymbol(d_charset_lens[i], &charset_lens[i], sizeof(int));
        }
        cudaMemcpyToSymbol(d_hashes, hashes.data(), hashes.size());
    }

    std::vector<std::string> run(uint64_t start, uint64_t end) {
        cudaMemset(d_count, 0, sizeof(int));
        launch_darkling(charset_ptrs.data(), charset_lens.data(), pwd_len,
                        hashes.data(), num_hashes, hash_len,
                        start, end, d_results, max_results, d_count,
                        grid, block);
        cudaDeviceSynchronize();
        int found = *h_count;
        std::vector<std::string> results;
        for (int i = 0; i < found && i < max_results; ++i) {
            results.emplace_back(h_results + i * MAX_MASK_LEN);
        }
        return results;
    }
};

int main() {
    DarklingContext ctx;
    ctx.allocate_buffers(10);
    std::vector<std::string> charsets = {"abc", "123"};
    std::vector<uint8_t> hashes(16); // placeholder
    ctx.preload(charsets, hashes, 2, 16);
    auto res = ctx.run(0, 1000);
    for (auto& s : res) std::cout << s << "\n";
    return 0;
}
