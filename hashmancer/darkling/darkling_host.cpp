#include "darkling_engine.h"
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
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
    uint8_t* d_all_hashes = nullptr;
    size_t d_all_hashes_size = 0;

    std::vector<std::string> prev_charsets;
    std::vector<uint8_t> prev_hashes;
    bool charsets_dirty = true;
    bool hashes_dirty = true;

    char* h_results = nullptr;
    char* d_results = nullptr;
    int* h_count = nullptr;
    int* d_count = nullptr;

    dim3 grid{128};
    dim3 block{256};

    bool tuned = false;

    void cleanup() {
        if (d_all_hashes) {
            cudaFree(d_all_hashes);
            d_all_hashes = nullptr;
            d_all_hashes_size = 0;
        }
        if (h_results) {
            cudaFreeHost(h_results);
            h_results = nullptr;
            d_results = nullptr;
        }
        if (h_count) {
            cudaFreeHost(h_count);
            h_count = nullptr;
            d_count = nullptr;
        }
    }

    ~DarklingContext() { cleanup(); }

    void upload_device_data(int start_hash, int batch_size) {
        cudaMemcpyToSymbol(d_pwd_len, &pwd_len, sizeof(int));
        cudaMemcpyToSymbol(d_hash_len, &hash_len, sizeof(int));
        cudaMemcpyToSymbol(d_num_hashes, &batch_size, sizeof(int));
        cudaMemcpyToSymbol(d_pos_charset, pos_map.data(), pwd_len);

        if(charsets_dirty) {
            for(int i=0;i<MAX_CUSTOM_SETS;i++) {
                int size = i < (int)charset_sizes.size() ? charset_sizes[i] : 0;
                cudaMemcpyToSymbol(d_charset_lens, &size, sizeof(int), i*sizeof(int));
                if(size > 0) {
                    const uint8_t* bp = i < (int)charset_byte_ptrs.size() ? charset_byte_ptrs[i] : nullptr;
                    const uint8_t* lp = i < (int)charset_len_ptrs.size() ? charset_len_ptrs[i] : nullptr;
                    cudaMemcpyToSymbol(d_charset_bytes, bp, size*MAX_UTF8_BYTES,
                                       i*MAX_CHARSET_CHARS*MAX_UTF8_BYTES);
                    cudaMemcpyToSymbol(d_charset_charlen, lp, size,
                                       i*MAX_CHARSET_CHARS);
                }
            }
            charsets_dirty = false;
        }

        if(hashes_dirty) {
            cudaMemcpy(d_all_hashes, hashes.data(), d_all_hashes_size, cudaMemcpyHostToDevice);
            hashes_dirty = false;
        }

        const uint8_t* slice = d_all_hashes + static_cast<size_t>(start_hash) * hash_len;
        cudaMemcpyToSymbol(d_hashes, slice, static_cast<size_t>(batch_size) * hash_len,
                           0, cudaMemcpyDeviceToDevice);
    }

    static void apply_power_limit(int device) {
        const char* lim = std::getenv("DARKLING_GPU_POWER_LIMIT");
        if(!lim) return;
        std::string cmd = "nvidia-smi -i " + std::to_string(device) + " -pl " + lim + " > /dev/null 2>&1";
        if (std::system(cmd.c_str()) != 0) {
            cmd = "rocm-smi -d " + std::to_string(device) + " --setpowerlimit " + lim + " > /dev/null 2>&1";
            std::system(cmd.c_str());
        }
    }

    void autotune(uint64_t sample_start, uint64_t sample_end) {
        const char* env = std::getenv("DARKLING_AUTOTUNE");
        if(env && std::string(env)=="0") return;
        int device = 0;
        cudaGetDevice(&device);

        const char* grid_env = std::getenv("DARKLING_GRID");
        const char* block_env = std::getenv("DARKLING_BLOCK");
        if(grid_env || block_env) {
            if(grid_env) grid.x = std::atoi(grid_env);
            if(block_env) block.x = std::atoi(block_env);
            apply_power_limit(device);
            tuned = true;
            return;
        }

        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, device);

        int maxThreads = prop.maxThreadsPerBlock;
        int active = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, crack_kernel, maxThreads, 0);
        grid.x = active * prop.multiProcessorCount;
        block.x = maxThreads;

        apply_power_limit(device);

        cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
        int batch = std::min(num_hashes, MAX_HASHES);
        upload_device_data(0, batch);
        cudaMemset(d_count, 0, sizeof(int));
        cudaEventRecord(s);
        launch_darkling_kernel(sample_start, sample_end, d_results, max_results, d_count, grid, block);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms=0; cudaEventElapsedTime(&ms, s, e);
        float rate = (float)(sample_end - sample_start)/(ms/1000.0f);
        std::cerr << "[darkling] initial speed " << rate << " H/s\n";
        tuned = true;
    }

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

        charsets_dirty = (prev_charsets != charsets);
        hashes_dirty = (prev_hashes != hsh);
        prev_charsets = charsets;
        prev_hashes = hsh;

        hashes = hsh;

        size_t required = static_cast<size_t>(num_hashes) * hash_len;
        if (d_all_hashes_size != required) {
            if (d_all_hashes) cudaFree(d_all_hashes);
            cudaMalloc(&d_all_hashes, required);
            d_all_hashes_size = required;
        }
        cudaMemcpy(d_all_hashes, hashes.data(), required, cudaMemcpyHostToDevice);
        hashes_dirty = false;

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
        if(!tuned) {
            uint64_t sample_end = start + std::min<uint64_t>(1000, end-start);
            autotune(start, sample_end);
        }
        std::vector<std::string> results;
        size_t result_off = 0;
        int remaining = max_results;
        for(int offset=0; offset < num_hashes && remaining > 0; offset += MAX_HASHES) {
            int batch = std::min(num_hashes - offset, MAX_HASHES);
            upload_device_data(offset, batch);
            cudaMemset(d_count, 0, sizeof(int));
            char* d_res_ptr = d_results + result_off * MAX_PWD_BYTES;
            launch_darkling_kernel(start, end, d_res_ptr, remaining, d_count,
                                   grid, block);
            cudaDeviceSynchronize();
            int found = *h_count;
            char* h_ptr = h_results + result_off * MAX_PWD_BYTES;
            for (int i = 0; i < found && i < remaining; ++i) {
                results.emplace_back(h_ptr + i * MAX_PWD_BYTES);
            }
            remaining = max_results - static_cast<int>(results.size());
            result_off = results.size();
        }
        return results;
    }
};

int main(int argc, char** argv) {
    uint64_t start = 0;
    uint64_t end = 1000;
    std::string cs_args[16];
    std::vector<std::string> hash_hex;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--start") == 0 && i + 1 < argc) {
            start = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--end") == 0 && i + 1 < argc) {
            end = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--hash") == 0 && i + 1 < argc) {
            hash_hex.push_back(argv[++i]);
        } else if (argv[i][0] == '-' && std::isdigit(argv[i][1]) &&
                   argv[i][2] == '\0' && i + 1 < argc) {
            int id = argv[i][1] - '1';
            if (id >= 0 && id < 16) {
                cs_args[id] = argv[++i];
            }
        }
    }

    DarklingContext ctx;
    ctx.allocate_buffers(10);

    std::vector<std::string> charsets;
    if (!cs_args[0].empty()) charsets.push_back(cs_args[0]);
    else charsets.push_back("abc");
    if (!cs_args[1].empty()) charsets.push_back(cs_args[1]);
    else charsets.push_back("123");

    std::vector<uint8_t> pos_map = {0,1};

    if (hash_hex.empty()) {
        std::cerr << "No --hash arguments provided" << std::endl;
        return 1;
    }

    size_t hlen = hash_hex[0].size();
    if (hlen % 2 != 0) {
        std::cerr << "Invalid hash length: " << hash_hex[0] << std::endl;
        return 1;
    }
    hlen /= 2;

    std::vector<uint8_t> hashes;
    for (const auto &h : hash_hex) {
        if (h.size() != hash_hex[0].size()) {
            std::cerr << "Hash length mismatch: " << h << std::endl;
            return 1;
        }
        for (size_t j = 0; j < h.size(); j += 2) {
            unsigned int val = std::stoi(h.substr(j, 2), nullptr, 16);
            hashes.push_back(static_cast<uint8_t>(val));
        }
    }

    ctx.preload(charsets, pos_map, hashes, 2, static_cast<int>(hlen));
    auto res = ctx.run(start, end);
    for (auto& s : res) std::cout << s << "\n";
    return 0;
}
