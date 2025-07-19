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

    std::vector<const uint8_t*> charset_byte_ptrs;
    std::vector<const uint8_t*> charset_len_ptrs;
    std::vector<int> charset_sizes;
    std::vector<std::vector<uint8_t>> charset_bytes;
    std::vector<std::vector<uint8_t>> charset_lens;
    std::vector<uint8_t> pos_map;
    std::vector<uint8_t> hashes;

    char* h_results = nullptr;
    char* d_results = nullptr;
    int* h_count = nullptr;
    int* d_count = nullptr;

    dim3 grid{128};
    dim3 block{256};

    void allocate_buffers(int max_res) {
        max_results = max_res;
        size_t res_size = static_cast<size_t>(max_results) * MAX_PWD_BYTES;
        cudaHostAlloc(&h_results, res_size, cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_results, h_results, 0);
        cudaHostAlloc(&h_count, sizeof(int), cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_count, h_count, 0);
    }

    void preload(const std::vector<std::string>& charsets,
                 const std::vector<uint8_t>& position_map,
                 const std::vector<uint8_t>& hsh,
                 int plen, int hlen) {
        pwd_len = plen;
        hash_len = hlen;
        num_hashes = hsh.size() / hlen;
        hashes = hsh;

        charset_byte_ptrs.clear();
        charset_len_ptrs.clear();
        charset_sizes.clear();
        charset_bytes.clear();
        charset_lens.clear();
        for (const auto& cs : charsets) {
            std::vector<uint8_t> bytes;
            std::vector<uint8_t> lens_vec;
            for (size_t i=0;i<cs.size();) {
                uint8_t b = static_cast<uint8_t>(cs[i]);
                int clen = 1;
                if ((b & 0x80)==0) clen = 1;
                else if ((b & 0xE0)==0xC0) clen = 2;
                else if ((b & 0xF0)==0xE0) clen = 3;
                else clen = 4;
                for (int j=0;j<clen;j++) bytes.push_back(static_cast<uint8_t>(cs[i+j]));
                for (int j=clen;j<MAX_UTF8_BYTES;j++) bytes.push_back(0);
                lens_vec.push_back(static_cast<uint8_t>(clen));
                i += clen;
            }
            charset_sizes.push_back(static_cast<int>(lens_vec.size()));
            charset_bytes.push_back(std::move(bytes));
            charset_lens.push_back(std::move(lens_vec));
        }
        for(size_t i=0;i<charset_bytes.size();++i){
            charset_byte_ptrs.push_back(charset_bytes[i].data());
            charset_len_ptrs.push_back(charset_lens[i].data());
        }
        pos_map = position_map;


    }

    std::vector<std::string> run(uint64_t start, uint64_t end) {
        cudaMemset(d_count, 0, sizeof(int));
        launch_darkling(charset_byte_ptrs.data(), charset_len_ptrs.data(),
                        charset_sizes.data(), pos_map.data(), pwd_len,
                        hashes.data(), num_hashes, hash_len,
                        start, end, d_results, max_results, d_count,
                        grid, block);
        cudaDeviceSynchronize();
        int found = *h_count;
        std::vector<std::string> results;
        for (int i = 0; i < found && i < max_results; ++i) {
            results.emplace_back(h_results + i * MAX_PWD_BYTES);
        }
        return results;
    }
};

int main() {
    DarklingContext ctx;
    ctx.allocate_buffers(10);
    std::vector<std::string> charsets = {"abc", "123"};
    std::vector<uint8_t> pos_map = {0,1};
    std::vector<uint8_t> hashes(16); // placeholder
    ctx.preload(charsets, pos_map, hashes, 2, 16);
    auto res = ctx.run(0, 1000);
    for (auto& s : res) std::cout << s << "\n";
    return 0;
}
