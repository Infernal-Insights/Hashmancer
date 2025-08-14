#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "darkling_rules.h"
#include "darkling_device_queue.h"
#include "darkling_telemetry.h"
#include "darkling_rule_manager.h"
#include "hash_primitives.cuh"
#include "gpu_manager.h"
#include "rule_analytics.h"
#include "checkpoint_manager.h"

// Minimal constants for mask kernel
#define MAX_UTF8_BYTES 4
#define MAX_CUSTOM_SETS 16
#define MAX_CHARSET_CHARS 256
#define MAX_MASK_LEN 32

extern "C" void load_darkling_data(const uint8_t **charset_bytes,
                                    const uint8_t **charset_lens,
                                    const int *charset_sizes,
                                    const uint8_t *pos_map, int pwd_len,
                                    const uint8_t *hashes, int num_hashes, int hash_len,
                                    uint8_t hash_type);

extern "C" void launch_darkling_kernel(uint64_t start, uint64_t end,
                                        char *d_results, int max_results, int *d_count,
                                        dim3 grid, dim3 block);

void launch_persistent(const uint8_t*, const uint32_t*, DlTelemetry*);

// -----------------------------------------------------------------------------
// Dictionary-only kernel used for attack 1
// -----------------------------------------------------------------------------
__global__ void dict_only_kernel(const uint8_t* words, const uint32_t* offsets,
                                 DlTelemetry* tel) {
  DlWorkItem item;
  while (dq_pop(&item)) {
    if (item.word_count == 0) continue;
    for (uint32_t i = 0; i < item.word_count; ++i) {
      uint32_t idx = item.word_start + i;
      const uint8_t* src = words + offsets[idx];
      uint32_t len = offsets[idx+1] - offsets[idx] - 1; // strip newline
      uint32_t dig[4];
      md5_hash(src, len, dig);
      atomicAdd(&tel->candidates_generated, 1ULL);
      atomicAdd(&tel->words_processed, 1ULL);
    }
  }
}

static void launch_dict_only(const uint8_t* d_words, const uint32_t* d_offsets,
                             DlTelemetry* tel) {
  dict_only_kernel<<<1,32>>>(d_words, d_offsets, tel);
}

// -----------------------------------------------------------------------------
// Attack helpers
// -----------------------------------------------------------------------------
static int run_mask_attack(const char* mask, OutputManager* output_mgr = nullptr, StatusReporter* status = nullptr) {
  if (!mask) {
    std::fprintf(stderr, "--mask required for attack 3\n");
    return 1;
  }
  // support numeric masks using built-in digit charset
  int mlen = 0;
  for (const char* p = mask; *p; ++p) {
    if (*p == '?') {
      if (*(p+1)) ++p; // skip charset id
    }
    ++mlen;
  }
  if (mlen <= 0 || mlen > MAX_MASK_LEN) {
    std::fprintf(stderr, "invalid mask length\n");
    return 1;
  }
  uint8_t digit_bytes[10][MAX_UTF8_BYTES];
  uint8_t digit_lens[10];
  for (int i = 0; i < 10; ++i) {
    digit_bytes[i][0] = '0' + i;
    for (int j = 1; j < MAX_UTF8_BYTES; ++j) digit_bytes[i][j] = 0;
    digit_lens[i] = 1;
  }
  const uint8_t* cs_bytes[MAX_CUSTOM_SETS];
  const uint8_t* cs_lens[MAX_CUSTOM_SETS];
  int cs_sizes[MAX_CUSTOM_SETS] = {0};
  cs_bytes[0] = (const uint8_t*)digit_bytes;
  cs_lens[0] = digit_lens;
  cs_sizes[0] = 10;
  uint8_t pos_map[MAX_MASK_LEN];
  for (int i = 0; i < mlen; ++i) pos_map[i] = 0;
  uint8_t dummy_hash[16] = {0};
  char* d_results; int* d_count;
  cudaMalloc(&d_results, 1);
  cudaMalloc(&d_count, sizeof(int));
  load_darkling_data(cs_bytes, cs_lens, cs_sizes, pos_map, mlen,
                     dummy_hash, 0, 16, 1);
  uint64_t end = 1; for (int i = 0; i < mlen; ++i) end *= 10ULL;
  dim3 grid(1), block(1);
  launch_darkling_kernel(0, end, d_results, 0, d_count, grid, block);
  cudaDeviceSynchronize();
  cudaFree(d_results); cudaFree(d_count);
  std::printf("mask attack generated %llu candidates\n",
              (unsigned long long)end);
  return 0;
}

static int run_dict_only(const char* shard, OutputManager* output_mgr = nullptr, StatusReporter* status = nullptr) {
  if (shard && strlen(shard) > 0) {
    std::fprintf(stderr, "Shard loading not yet implemented, using test data\n");
  }
  const char* words = "hello\nworld\n";
  uint32_t offsets_host[3] = {0,6,12};
  uint8_t* d_words; uint32_t* d_offsets;
  cudaMalloc(&d_words, 12);
  cudaMemcpy(d_words, words, 12, cudaMemcpyHostToDevice);
  cudaMalloc(&d_offsets, sizeof(offsets_host));
  cudaMemcpy(d_offsets, offsets_host, sizeof(offsets_host), cudaMemcpyHostToDevice);
  DlTelemetry* d_tel; cudaMalloc(&d_tel, sizeof(DlTelemetry));
  cudaMemset(d_tel, 0, sizeof(DlTelemetry));
  DlWorkItem item{0,2,0,0};
  uint32_t zero = 0, tail = 1;
  cudaMemcpyToSymbol(g_qhead, &zero, sizeof(uint32_t));
  cudaMemcpyToSymbol(g_qtail, &zero, sizeof(uint32_t));
  cudaMemcpyToSymbol(g_queue, &item, sizeof(DlWorkItem));
  cudaMemcpyToSymbol(g_qtail, &tail, sizeof(uint32_t));
  launch_dict_only(d_words, d_offsets, d_tel);
  cudaDeviceSynchronize();
  DlTelemetry h{}; cudaMemcpy(&h, d_tel, sizeof(h), cudaMemcpyDeviceToHost);
  std::printf("words_processed=%llu candidates=%llu\n",
              (unsigned long long)h.words_processed,
              (unsigned long long)h.candidates_generated);
  cudaFree(d_tel); cudaFree(d_words); cudaFree(d_offsets);
  return 0;
}

static int run_dict_rules(const char* shard, const char* ruleset, DlRuleManager* manager, 
                          OutputManager* output_mgr = nullptr, StatusReporter* status = nullptr) {
  // Load shard file or use default words for testing
  const char* words = "hello\nworld\ntest\npassword\n";
  uint32_t offsets_host[5] = {0, 6, 12, 17, 26};
  
  if (shard && strlen(shard) > 0) {
    // TODO: Implement real shard loading
    std::fprintf(stderr, "Shard loading not yet implemented, using test data\n");
  }
  
  // Initialize GPU memory
  uint8_t* d_words; 
  uint32_t* d_offsets;
  cudaMalloc(&d_words, 26);
  cudaMemcpy(d_words, words, 26, cudaMemcpyHostToDevice);
  cudaMalloc(&d_offsets, sizeof(offsets_host));
  cudaMemcpy(d_offsets, offsets_host, sizeof(offsets_host), cudaMemcpyHostToDevice);
  
  DlTelemetry* d_tel; 
  cudaMalloc(&d_tel, sizeof(DlTelemetry));
  cudaMemset(d_tel, 0, sizeof(DlTelemetry));
  
  // Set up work item for rule-based processing
  DlWorkItem item{0, 4, 0, 2}; // 4 words, rule start at 0
  uint32_t zero = 0, tail = 1;
  cudaMemcpyToSymbol(g_qhead, &zero, sizeof(uint32_t));
  cudaMemcpyToSymbol(g_qtail, &zero, sizeof(uint32_t));
  cudaMemcpyToSymbol(g_queue, &item, sizeof(DlWorkItem));
  cudaMemcpyToSymbol(g_qtail, &tail, sizeof(uint32_t));
  
  // Get built-in rule set from manager
  const DlRuleSet* builtin_rules = dl_get_builtin_best64(manager);
  if (builtin_rules && builtin_rules->rule_count > 0) {
    std::printf("Using %u built-in rules\n", builtin_rules->rule_count);
    
    // Execute rules using new rule manager
    uint32_t word_count = 4;
    uint32_t max_output_length = 64;
    
    // Allocate output buffers
    uint8_t* output_candidates = new uint8_t[word_count * builtin_rules->rule_count * max_output_length];
    uint32_t* output_lengths = new uint32_t[word_count * builtin_rules->rule_count];
    
    // Copy input words to host format for rule manager
    uint8_t input_words[word_count * max_output_length];
    uint32_t input_lengths[word_count];
    memset(input_words, 0, sizeof(input_words));
    
    // Copy words from GPU format to rule manager format
    const char* word_ptrs[4] = {"hello", "world", "test", "password"};
    for (uint32_t i = 0; i < word_count; ++i) {
      const char* word = word_ptrs[i];
      uint32_t len = strlen(word);
      input_lengths[i] = len;
      memcpy(input_words + i * max_output_length, word, len);
    }
    
    // Execute rules
    dl_execute_rule_batch_gpu(builtin_rules, input_words, input_lengths,
                              output_candidates, output_lengths,
                              word_count, max_output_length);
    
    // Count successful rule applications
    uint64_t candidates_generated = 0;
    for (uint32_t i = 0; i < word_count * builtin_rules->rule_count; ++i) {
      if (output_lengths[i] > 0) {
        candidates_generated++;
      }
    }
    
    std::printf("Rule manager processed %u words with %u rules, generated %llu candidates\n",
                word_count, builtin_rules->rule_count, (unsigned long long)candidates_generated);
    
    delete[] output_candidates;
    delete[] output_lengths;
  } else {
    // Fallback to legacy rule processing
    if (ruleset && strlen(ruleset) > 0) {
      dl_rules_load_json(ruleset);
    }
    
    launch_persistent(d_words, d_offsets, d_tel);
    cudaDeviceSynchronize();
    
    DlTelemetry h{}; 
    cudaMemcpy(&h, d_tel, sizeof(h), cudaMemcpyDeviceToHost);
    std::printf("Legacy rules: words_processed=%llu candidates=%llu\n",
                (unsigned long long)h.words_processed,
                (unsigned long long)h.candidates_generated);
  }
  
  cudaFree(d_tel); 
  cudaFree(d_words); 
  cudaFree(d_offsets);
  return 0;
}

static void run_external_rules(const char* shard, const char* rulefile) {
  (void)shard; (void)rulefile;
  std::fprintf(stderr, "attack 3 (external rules) not implemented\n");
}

// Command line arguments structure
struct CLIArgs {
    int hash_mode = 0;              // -m, --hash-type
    int attack_mode = -1;           // -a, --attack-mode
    std::string hash_file;          // hash file path
    std::string wordlist;           // wordlist/dictionary file
    std::string mask;               // mask for attack mode 3
    std::vector<std::string> rules; // -r, --rules (can be multiple)
    std::vector<std::string> shards; // --shard (can be multiple)
    std::string outfile;            // --outfile
    int outfile_format = 2;         // --outfile-format
    std::string restore_file;       // --restore-file
    int device_id = 1;              // -d, --device
    bool quiet = false;             // --quiet
    bool status = false;            // --status
    bool status_json = false;       // --status-json
    int status_timer = 10;          // --status-timer
    uint64_t start = 0;             // --start (for ranges)
    uint64_t end = 0;               // --end (for ranges)
    bool optimized = false;         // -O, --optimized
    int workload = 2;               // -w, --workload-profile
    std::vector<std::string> custom_charsets; // -1, -2, -3, -4 (custom charsets)
    
    // Legacy Darkling options
    std::string ruleset;            // --ruleset (for backward compatibility)
    
    // Advanced performance options
    bool multi_gpu = false;         // --multi-gpu
    bool smart_rules = false;       // --smart-rules
    std::string checkpoint_file;    // --checkpoint
    bool resume = false;            // --resume
    int checkpoint_interval = 300;  // --checkpoint-interval (seconds)
    bool analytics = false;         // --analytics
    std::string job_id;             // --job-id
    bool benchmark = false;         // --benchmark
};

void print_usage(const char* prog) {
    std::printf("Usage: %s [options] hash_file [wordlist|mask]\n\n", prog);
    std::printf("Hash modes:\n");
    std::printf("  -m, --hash-type NUM        Hash type (0=MD5, 100=SHA1, 1000=NTLM)\n\n");
    std::printf("Attack modes:\n");
    std::printf("  -a, --attack-mode NUM      Attack mode (0=dict, 3=mask)\n\n");
    std::printf("Options:\n");
    std::printf("  -r, --rules FILE           Rules file (hashcat format)\n");
    std::printf("  --shard FILE               Dictionary shard file\n");
    std::printf("  --outfile FILE             Output file for cracked hashes\n");
    std::printf("  --outfile-format NUM       Output format (2=plain)\n");
    std::printf("  --restore-file FILE        Restore file path\n");
    std::printf("  -d, --device NUM           GPU device ID\n");
    std::printf("  --quiet                    Suppress output\n");
    std::printf("  --status                   Enable status updates\n");
    std::printf("  --status-json              Status in JSON format\n");
    std::printf("  --status-timer NUM         Status timer interval\n");
    std::printf("  --start NUM                Start offset\n");
    std::printf("  --end NUM                  End offset\n");
    std::printf("  -O, --optimized            Enable optimized kernels\n");
    std::printf("  -w, --workload-profile NUM Workload profile (1-4)\n");
    std::printf("  -1, -2, -3, -4 CHARSET     Custom charset\n");
    std::printf("  --ruleset FILE             Legacy ruleset JSON file\n");
    std::printf("\n");
    std::printf("Advanced Performance Options:\n");
    std::printf("  --multi-gpu                Enable multi-GPU acceleration\n");
    std::printf("  --smart-rules              Use AI-powered rule selection\n");
    std::printf("  --checkpoint FILE          Checkpoint file for resume\n");
    std::printf("  --resume                   Resume from checkpoint\n");
    std::printf("  --checkpoint-interval SEC  Auto-checkpoint interval\n");
    std::printf("  --analytics                Enable rule analytics\n");
    std::printf("  --job-id ID                Unique job identifier\n");
    std::printf("  --benchmark                Run performance benchmark\n");
    std::printf("  -h, --help                 Show this help\n");
}

bool parse_args(int argc, char** argv, CLIArgs& args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return false;
        }
        else if ((arg == "-m" || arg == "--hash-type") && i + 1 < argc) {
            args.hash_mode = std::atoi(argv[++i]);
        }
        else if ((arg == "-a" || arg == "--attack-mode") && i + 1 < argc) {
            args.attack_mode = std::atoi(argv[++i]);
        }
        else if ((arg == "-r" || arg == "--rules") && i + 1 < argc) {
            args.rules.push_back(argv[++i]);
        }
        else if (arg == "--shard" && i + 1 < argc) {
            args.shards.push_back(argv[++i]);
        }
        else if (arg == "--outfile" && i + 1 < argc) {
            args.outfile = argv[++i];
        }
        else if (arg == "--outfile-format" && i + 1 < argc) {
            args.outfile_format = std::atoi(argv[++i]);
        }
        else if (arg == "--restore-file" && i + 1 < argc) {
            args.restore_file = argv[++i];
        }
        else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            args.device_id = std::atoi(argv[++i]);
        }
        else if (arg == "--quiet") {
            args.quiet = true;
        }
        else if (arg == "--status") {
            args.status = true;
        }
        else if (arg == "--status-json") {
            args.status_json = true;
        }
        else if (arg == "--status-timer" && i + 1 < argc) {
            args.status_timer = std::atoi(argv[++i]);
        }
        else if (arg == "--start" && i + 1 < argc) {
            args.start = std::stoull(argv[++i]);
        }
        else if (arg == "--end" && i + 1 < argc) {
            args.end = std::stoull(argv[++i]);
        }
        else if (arg == "-O" || arg == "--optimized") {
            args.optimized = true;
        }
        else if ((arg == "-w" || arg == "--workload-profile") && i + 1 < argc) {
            args.workload = std::atoi(argv[++i]);
        }
        else if ((arg == "-1" || arg == "-2" || arg == "-3" || arg == "-4") && i + 1 < argc) {
            args.custom_charsets.push_back(std::string(arg.substr(1)) + ":" + argv[++i]);
        }
        // Legacy support
        else if (arg == "--attack" && i + 1 < argc) {
            args.attack_mode = std::atoi(argv[++i]);
        }
        else if (arg == "--ruleset" && i + 1 < argc) {
            args.ruleset = argv[++i];
        }
        else if (arg == "--mask" && i + 1 < argc) {
            args.mask = argv[++i];
        }
        // Advanced performance options
        else if (arg == "--multi-gpu") {
            args.multi_gpu = true;
        }
        else if (arg == "--smart-rules") {
            args.smart_rules = true;
        }
        else if (arg == "--checkpoint" && i + 1 < argc) {
            args.checkpoint_file = argv[++i];
        }
        else if (arg == "--resume") {
            args.resume = true;
        }
        else if (arg == "--checkpoint-interval" && i + 1 < argc) {
            args.checkpoint_interval = std::atoi(argv[++i]);
        }
        else if (arg == "--analytics") {
            args.analytics = true;
        }
        else if (arg == "--job-id" && i + 1 < argc) {
            args.job_id = argv[++i];
        }
        else if (arg == "--benchmark") {
            args.benchmark = true;
        }
        // Positional arguments
        else if (arg[0] != '-') {
            if (args.hash_file.empty()) {
                args.hash_file = arg;
            } else if (args.wordlist.empty() && args.mask.empty()) {
                if (args.attack_mode == 3) {
                    args.mask = arg;
                } else {
                    args.wordlist = arg;
                }
            }
        }
        else {
            std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            return false;
        }
    }
    
    // Validate required arguments
    if (args.hash_file.empty() && args.attack_mode != -1) {
        std::fprintf(stderr, "Hash file required\n");
        return false;
    }
    
    // Infer attack mode if not specified
    if (args.attack_mode == -1) {
        if (!args.mask.empty()) {
            args.attack_mode = 3; // mask attack
        } else if (!args.wordlist.empty() || !args.shards.empty()) {
            args.attack_mode = 0; // dictionary attack
        } else {
            std::fprintf(stderr, "Attack mode could not be determined\n");
            return false;
        }
    }
    
    return true;
}

// Convert hashcat rule file to internal format
bool convert_hashcat_rules(const std::string& rule_file, DlRuleManager* manager) {
    // First try the new hashcat rule parser
    DlRuleSet parsed_rules;
    if (dl_parse_hashcat_rules(rule_file.c_str(), &parsed_rules, "hashcat_rules")) {
        std::printf("Parsed %u hashcat rules from %s\n", parsed_rules.rule_count, rule_file.c_str());
        
        // Add the parsed rule set to the manager
        // For now, we'll use the existing load function as fallback
        // TODO: Add proper rule set management to the rule manager
        return true;
    } else {
        // Fallback to loading as user rules
        if (!dl_load_user_rules_from_file(manager, rule_file.c_str(), "hashcat_rules")) {
            std::fprintf(stderr, "Failed to load rules from %s\n", rule_file.c_str());
            return false;
        }
        return true;
    }
}

// Hash parsing utilities
bool is_hex_string(const std::string& str) {
    return std::all_of(str.begin(), str.end(), [](char c) {
        return std::isxdigit(c);
    });
}

uint32_t hex_to_uint32(const std::string& hex_str, size_t offset) {
    if (offset + 8 > hex_str.length()) return 0;
    
    uint32_t result = 0;
    for (size_t i = 0; i < 8; ++i) {
        char c = hex_str[offset + i];
        uint32_t digit;
        if (c >= '0' && c <= '9') {
            digit = c - '0';
        } else if (c >= 'a' && c <= 'f') {
            digit = c - 'a' + 10;
        } else if (c >= 'A' && c <= 'F') {
            digit = c - 'A' + 10;
        } else {
            return 0; // Invalid hex
        }
        result = (result << 4) | digit;
    }
    return result;
}

struct HashInfo {
    std::vector<uint32_t> hash_data;
    int expected_length;
    std::string hash_name;
};

// Load and validate hash file
HashInfo load_hash_file(const std::string& hash_file, int hash_mode) {
    HashInfo result;
    
    // Set expected hash lengths and names based on mode
    switch (hash_mode) {
        case 0:    // MD5
            result.expected_length = 32;
            result.hash_name = "MD5";
            break;
        case 100:  // SHA1
            result.expected_length = 40;
            result.hash_name = "SHA1";
            break;
        case 1000: // NTLM
            result.expected_length = 32;
            result.hash_name = "NTLM";
            break;
        default:
            std::fprintf(stderr, "Unsupported hash mode: %d\n", hash_mode);
            return result;
    }
    
    std::ifstream file(hash_file);
    if (!file.is_open()) {
        std::fprintf(stderr, "Cannot open hash file: %s\n", hash_file.c_str());
        return result;
    }
    
    std::string line;
    int line_number = 0;
    int valid_hashes = 0;
    
    while (std::getline(file, line)) {
        line_number++;
        
        // Remove whitespace
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        // Convert to lowercase for consistent processing
        std::transform(line.begin(), line.end(), line.begin(), ::tolower);
        
        // Extract hash part (before any colon for hash:salt format)
        size_t colon_pos = line.find(':');
        std::string hash_part = (colon_pos != std::string::npos) ? 
                               line.substr(0, colon_pos) : line;
        
        // Validate hash length and format
        if (hash_part.length() != static_cast<size_t>(result.expected_length)) {
            std::fprintf(stderr, "Warning: Invalid %s hash length at line %d (expected %d, got %zu)\n",
                        result.hash_name.c_str(), line_number, result.expected_length, hash_part.length());
            continue;
        }
        
        if (!is_hex_string(hash_part)) {
            std::fprintf(stderr, "Warning: Invalid hex characters in hash at line %d\n", line_number);
            continue;
        }
        
        // Convert hex string to uint32_t array
        if (hash_mode == 0 || hash_mode == 1000) { // MD5 or NTLM (128-bit)
            for (int i = 0; i < 4; ++i) {
                uint32_t value = hex_to_uint32(hash_part, i * 8);
                result.hash_data.push_back(value);
            }
        } else if (hash_mode == 100) { // SHA1 (160-bit)
            for (int i = 0; i < 5; ++i) {
                uint32_t value = hex_to_uint32(hash_part, i * 8);
                result.hash_data.push_back(value);
            }
        }
        
        valid_hashes++;
    }
    
    file.close();
    
    std::printf("Loaded %d valid %s hashes from %s\n", 
               valid_hashes, result.hash_name.c_str(), hash_file.c_str());
    
    return result;
}

// Output file management
class OutputManager {
private:
    std::string outfile_path;
    int format;
    std::ofstream output_stream;
    
public:
    OutputManager(const std::string& path, int fmt) : outfile_path(path), format(fmt) {
        if (!path.empty()) {
            output_stream.open(path, std::ios::app); // Append mode for results
            if (!output_stream.is_open()) {
                std::fprintf(stderr, "Warning: Cannot open output file %s\n", path.c_str());
            }
        }
    }
    
    ~OutputManager() {
        if (output_stream.is_open()) {
            output_stream.close();
        }
    }
    
    void write_found(const std::string& plaintext, const std::string& hash = "") {
        std::string output;
        
        switch (format) {
            case 1: // hash:plain
                output = hash + ":" + plaintext;
                break;
            case 2: // plain (default)
                output = plaintext;
                break;
            case 3: // hex_plain
                output = "";
                for (char c : plaintext) {
                    char hex_buf[3];
                    sprintf(hex_buf, "%02x", static_cast<unsigned char>(c));
                    output += hex_buf;
                }
                break;
            default:
                output = plaintext;
                break;
        }
        
        // Write to file if available
        if (output_stream.is_open()) {
            output_stream << output << std::endl;
            output_stream.flush();
        }
        
        // Always write to stdout
        std::printf("%s\n", output.c_str());
    }
    
    bool is_valid() const {
        return outfile_path.empty() || output_stream.is_open();
    }
};

// Status reporting for hashcat compatibility
class StatusReporter {
private:
    bool enable_status;
    bool json_format;
    int timer_interval;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point last_status;
    
public:
    StatusReporter(bool status, bool json, int interval) 
        : enable_status(status), json_format(json), timer_interval(interval) {
        start_time = std::chrono::steady_clock::now();
        last_status = start_time;
    }
    
    void report_status(uint64_t candidates_tested, uint64_t total_candidates, 
                      double hashrate, uint32_t found_count) {
        if (!enable_status) return;
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_status);
        
        if (elapsed.count() < timer_interval) return;
        
        last_status = now;
        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
        
        double progress = total_candidates > 0 ? 
            static_cast<double>(candidates_tested) / total_candidates * 100.0 : 0.0;
        
        if (json_format) {
            std::printf("{\"status\":\"running\",\"progress\":[%.2f],\"speed\":[%.0f],"
                       "\"time\":%ld,\"found\":%u}\n",
                       progress, hashrate, total_elapsed.count(), found_count);
        } else {
            std::printf("Status.......: Running\n");
            std::printf("Progress.....: %.2f%% (%llu/%llu)\n", 
                       progress, candidates_tested, total_candidates);
            std::printf("Speed........: %.0f H/s\n", hashrate);
            std::printf("Time.........: %ld seconds\n", total_elapsed.count());
            std::printf("Found........: %u\n\n", found_count);
        }
        std::fflush(stdout);
    }
    
    void final_status(uint64_t total_tested, uint32_t found_count) {
        auto now = std::chrono::steady_clock::now();
        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
        
        if (json_format) {
            std::printf("{\"status\":\"exhausted\",\"progress\":[100.0],\"time\":%ld,\"found\":%u}\n",
                       total_elapsed.count(), found_count);
        } else {
            std::printf("Status.......: Exhausted\n");
            std::printf("Tested.......: %llu\n", total_tested);
            std::printf("Found........: %u\n", found_count);
            std::printf("Time.........: %ld seconds\n", total_elapsed.count());
        }
    }
};

int main(int argc, char** argv) {
    CLIArgs args;
    
    if (!parse_args(argc, argv, args)) {
        return 1;
    }
    
    // Initialize GPU Manager for advanced multi-GPU support
    std::unique_ptr<GPUManager> gpu_manager;
    std::vector<int> active_devices;
    
    if (args.multi_gpu || args.benchmark) {
        gpu_manager = std::make_unique<GPUManager>();
        if (!gpu_manager->initialize()) {
            std::fprintf(stderr, "Failed to initialize GPU manager\n");
            return 1;
        }
        
        auto available_gpus = gpu_manager->get_available_gpus();
        if (available_gpus.empty()) {
            std::fprintf(stderr, "No available GPUs found\n");
            return 1;
        }
        
        for (const auto& gpu : available_gpus) {
            active_devices.push_back(gpu.device_id);
        }
        
        if (!args.quiet) {
            std::printf("Multi-GPU mode: Using %zu GPUs\n", active_devices.size());
            for (const auto& gpu : available_gpus) {
                std::printf("  GPU %d: %s (%zu MB)\n", 
                           gpu.device_id, gpu.name.c_str(), 
                           gpu.total_memory / (1024 * 1024));
            }
        }
    } else {
        // Single GPU mode
        cudaError_t err = cudaSetDevice(args.device_id - 1);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "Failed to set CUDA device %d: %s\n", 
                        args.device_id, cudaGetErrorString(err));
            return 1;
        }
        active_devices.push_back(args.device_id - 1);
    }
    
    // Initialize rule manager
    DlRuleManager* rule_manager = dl_create_rule_manager();
    if (!rule_manager) {
        std::fprintf(stderr, "Failed to create rule manager\n");
        return 1;
    }
    
    // Load built-in rules
    if (!dl_load_builtin_rules(rule_manager)) {
        std::fprintf(stderr, "Failed to load built-in rules\n");
        dl_destroy_rule_manager(rule_manager);
        return 1;
    }
    
    if (!dl_load_ptx_rules(rule_manager)) {
        std::fprintf(stderr, "Warning: PTX rules not available\n");
    }
    
    // Initialize analytics system if requested
    std::unique_ptr<RuleAnalytics> analytics_system;
    std::unique_ptr<SmartRuleSelector> smart_selector;
    
    if (args.analytics || args.smart_rules) {
        analytics_system = std::make_unique<RuleAnalytics>();
        if (analytics_system->initialize()) {
            if (!args.quiet) {
                std::printf("Rule analytics system initialized\n");
            }
            
            if (args.smart_rules) {
                smart_selector = std::make_unique<SmartRuleSelector>(analytics_system.get());
                if (!args.quiet) {
                    std::printf("Smart rule selection enabled\n");
                }
            }
        } else {
            std::fprintf(stderr, "Warning: Failed to initialize analytics system\n");
        }
    }
    
    // Initialize checkpoint system
    std::unique_ptr<CheckpointManager> checkpoint_manager;
    std::unique_ptr<JobRecoverySystem> recovery_system;
    std::string effective_job_id = args.job_id;
    
    if (!args.checkpoint_file.empty() || args.resume || !effective_job_id.empty()) {
        checkpoint_manager = std::make_unique<CheckpointManager>();
        if (checkpoint_manager->initialize()) {
            checkpoint_manager->set_auto_checkpoint_interval(args.checkpoint_interval);
            recovery_system = std::make_unique<JobRecoverySystem>(checkpoint_manager.get());
            
            if (effective_job_id.empty()) {
                // Generate job ID based on parameters
                effective_job_id = "job_" + std::to_string(std::time(nullptr));
            }
            
            if (!args.quiet) {
                std::printf("Checkpoint system initialized (Job ID: %s)\n", effective_job_id.c_str());
            }
        } else {
            std::fprintf(stderr, "Warning: Failed to initialize checkpoint system\n");
        }
    }
    
    // Load user rules if specified
    for (const auto& rule_file : args.rules) {
        if (!convert_hashcat_rules(rule_file, rule_manager)) {
            dl_destroy_rule_manager(rule_manager);
            return 1;
        }
    }
    
    // Load legacy ruleset for backward compatibility
    if (!args.ruleset.empty()) {
        dl_rules_load_json(args.ruleset.c_str());
    }
    
    // Load hashes if hash file provided
    HashInfo hash_info;
    if (!args.hash_file.empty()) {
        hash_info = load_hash_file(args.hash_file, args.hash_mode);
        if (hash_info.hash_data.empty()) {
            std::fprintf(stderr, "No valid hashes loaded\n");
            dl_destroy_rule_manager(rule_manager);
            return 1;
        }
    }
    
    // Initialize output manager
    OutputManager output_mgr(args.outfile, args.outfile_format);
    if (!output_mgr.is_valid()) {
        std::fprintf(stderr, "Failed to initialize output file\n");
        dl_destroy_rule_manager(rule_manager);
        return 1;
    }
    
    // Initialize status reporter
    StatusReporter status_reporter(args.status || args.status_json, args.status_json, args.status_timer);
    
    if (!args.quiet) {
        std::printf("darkling-engine starting...\n");
        std::printf("Hash mode: %d (%s)\n", args.hash_mode, 
                   args.hash_mode == 0 ? "MD5" : 
                   args.hash_mode == 100 ? "SHA1" : 
                   args.hash_mode == 1000 ? "NTLM" : "Unknown");
        std::printf("Attack mode: %d (%s)\n", args.attack_mode,
                   args.attack_mode == 0 ? "Dictionary" :
                   args.attack_mode == 3 ? "Mask" : "Unknown");
        if (!hash_info.hash_data.empty()) {
            std::printf("Target hashes: %zu\n", hash_info.hash_data.size() / 
                       (args.hash_mode == 100 ? 5 : 4)); // SHA1 is 5 uint32s, others are 4
        }
        std::printf("\n");
    }
    
    // Execute attack based on mode
    int result = 0;
    switch (args.attack_mode) {
        case 0: // Dictionary attack
            if (!args.wordlist.empty()) {
                // Use wordlist
                if (!args.rules.empty()) {
                    result = run_dict_rules(args.wordlist.c_str(), "", rule_manager, &output_mgr, &status_reporter);
                } else {
                    result = run_dict_only(args.wordlist.c_str(), &output_mgr, &status_reporter);
                }
            } else if (!args.shards.empty()) {
                // Use shards
                result = run_dict_rules(args.shards[0].c_str(), "", rule_manager, &output_mgr, &status_reporter);
            } else {
                std::fprintf(stderr, "Dictionary attack requires wordlist or shard\n");
                result = 1;
            }
            break;
            
        case 3: // Mask attack
            if (!args.mask.empty()) {
                result = run_mask_attack(args.mask.c_str(), &output_mgr, &status_reporter);
            } else {
                std::fprintf(stderr, "Mask attack requires mask\n");
                result = 1;
            }
            break;
            
        default:
            std::fprintf(stderr, "Unsupported attack mode: %d\n", args.attack_mode);
            result = 1;
            break;
    }
    
    // Cleanup
    dl_destroy_rule_manager(rule_manager);
    return result;
}
