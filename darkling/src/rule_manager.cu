#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "darkling_rule_manager.h"
#include "hash_primitives.cuh"
#include "kernel_optimizations.cuh"

// Built-in Best64 rule strings
const char* DL_BEST64_RULES[64] = {
    ":",      // no-op
    "l",      // lowercase
    "u",      // uppercase  
    "c",      // capitalize
    "C",      // invert case
    "t",      // toggle case
    "r",      // reverse
    "d",      // duplicate
    "f",      // reflect
    "{",      // rotate left
    "}",      // rotate right
    "[",      // delete first
    "]",      // delete last
    "$0", "$1", "$2", "$3", "$4", "$5", "$6", "$7", "$8", "$9",  // append digits
    "^0", "^1", "^2", "^3", "^4", "^5", "^6", "^7", "^8", "^9",  // prepend digits
    "$!", "$@", "$#", "$$", "$%", "$^", "$&", "$*",              // append symbols
    "^!", "^@", "^#", "^$", "^%", "^^", "^&", "^*",              // prepend symbols
    "se3", "sa@", "si1", "so0", "sl1", "ss$",                    // leet substitutions
    "sA@", "sI1", "sS$", "sO0", "sL1",                           // leet upper
    "D0", "D1", "D2",                                            // delete at pos
    "]D0", "]D1", "]D2",                                         // delete last then pos
    "[c", "c]", "c$0", "c$1", "c$2"                              // capitalize combos
};

// Forward declarations for PTX functions (would be generated)
extern "C" {
    __device__ void rule_best64_0(uint8_t*, const uint8_t*, uint32_t, const DlRuleParams*, uint32_t);
    __device__ void rule_best64_1(uint8_t*, const uint8_t*, uint32_t, const DlRuleParams*, uint32_t);
    __device__ void rule_best64_2(uint8_t*, const uint8_t*, uint32_t, const DlRuleParams*, uint32_t);
    // ... more PTX function declarations would be auto-generated
}

// PTX function table (would be auto-generated)
static DlPTXRuleFunc ptx_best64_functions[64] = {
    rule_best64_0,  // :
    rule_best64_1,  // l  
    rule_best64_2,  // u
    // ... more entries
    nullptr         // Not all rules may have PTX implementations
};

// GPU kernel for hybrid rule execution (PTX + interpreted)
__global__ void hybrid_rule_execution_kernel(
    const uint8_t* input_words,        // [word_count * max_word_len]
    const uint32_t* input_lengths,     // [word_count]
    const DlCompiledRule* rules,       // [rule_count]
    uint8_t* output_candidates,        // [word_count * rule_count * max_word_len]
    uint32_t* output_lengths,          // [word_count * rule_count]
    uint32_t word_count,
    uint32_t rule_count,
    uint32_t max_word_len,
    DlTelemetry* telemetry) {
    
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    
    for (uint32_t i = tid; i < word_count * rule_count; i += stride) {
        uint32_t word_idx = i / rule_count;
        uint32_t rule_idx = i % rule_count;
        
        if (word_idx >= word_count || rule_idx >= rule_count) continue;
        
        const uint8_t* input_word = input_words + word_idx * max_word_len;
        uint32_t input_len = input_lengths[word_idx];
        DlCompiledRule rule = rules[rule_idx];
        
        uint8_t* output_candidate = output_candidates + i * max_word_len;
        uint32_t output_len = input_len;
        
        // Apply rule transformation
        if (rule.type == RULE_TYPE_BUILTIN_PTX && rule.rule_id < 64) {
            // Use optimized PTX function
            DlPTXRuleFunc ptx_func = ptx_best64_functions[rule.rule_id];
            if (ptx_func) {
                ptx_func(output_candidate, input_word, input_len, &rule.params, 0);
                output_len = input_len + rule.params.max_length_delta;
            } else {
                // Fallback to interpreted rule
                apply_interpreted_rule(output_candidate, input_word, input_len, &rule, &output_len);
            }
        } else {
            // Use interpreted rule implementation
            apply_interpreted_rule(output_candidate, input_word, input_len, &rule, &output_len);
        }
        
        output_lengths[i] = output_len;
        atomicAdd(&telemetry->candidates_generated, 1ULL);
    }
}

// Interpreted rule implementation for user-uploaded rules
__device__ void apply_interpreted_rule(uint8_t* output, const uint8_t* input, uint32_t input_len,
                                      const DlCompiledRule* rule, uint32_t* output_len) {
    // Parse and execute rule string at runtime
    const char* rule_str = rule->rule_string;
    
    if (strcmp(rule_str, ":") == 0) {
        // No-op: copy input to output
        for (uint32_t i = 0; i < input_len; ++i) {
            output[i] = input[i];
        }
        *output_len = input_len;
    }
    else if (strcmp(rule_str, "l") == 0) {
        // Lowercase
        for (uint32_t i = 0; i < input_len; ++i) {
            uint8_t c = input[i];
            if (c >= 'A' && c <= 'Z') {
                output[i] = c + 32;
            } else {
                output[i] = c;
            }
        }
        *output_len = input_len;
    }
    else if (strcmp(rule_str, "u") == 0) {
        // Uppercase
        for (uint32_t i = 0; i < input_len; ++i) {
            uint8_t c = input[i];
            if (c >= 'a' && c <= 'z') {
                output[i] = c - 32;
            } else {
                output[i] = c;
            }
        }
        *output_len = input_len;
    }
    else if (strcmp(rule_str, "r") == 0) {
        // Reverse
        for (uint32_t i = 0; i < input_len; ++i) {
            output[i] = input[input_len - 1 - i];
        }
        *output_len = input_len;
    }
    else if (strcmp(rule_str, "d") == 0) {
        // Duplicate
        for (uint32_t i = 0; i < input_len; ++i) {
            output[i] = input[i];
            output[i + input_len] = input[i];
        }
        *output_len = input_len * 2;
    }
    else if (rule_str[0] == '$' && strlen(rule_str) == 2) {
        // Append character
        for (uint32_t i = 0; i < input_len; ++i) {
            output[i] = input[i];
        }
        output[input_len] = rule_str[1];
        *output_len = input_len + 1;
    }
    else if (rule_str[0] == '^' && strlen(rule_str) == 2) {
        // Prepend character
        output[0] = rule_str[1];
        for (uint32_t i = 0; i < input_len; ++i) {
            output[i + 1] = input[i];
        }
        *output_len = input_len + 1;
    }
    else if (rule_str[0] == 's' && strlen(rule_str) == 3) {
        // Character substitution
        uint8_t from_char = rule_str[1];
        uint8_t to_char = rule_str[2];
        for (uint32_t i = 0; i < input_len; ++i) {
            output[i] = (input[i] == from_char) ? to_char : input[i];
        }
        *output_len = input_len;
    }
    else {
        // Unknown rule: copy input
        for (uint32_t i = 0; i < input_len; ++i) {
            output[i] = input[i];
        }
        *output_len = input_len;
    }
}

// Host implementation
class RuleManagerImpl {
public:
    DlRuleSet builtin_best64;
    
private:
    std::vector<DlRuleSet> user_rule_sets;
    DlRuleFunctionTable function_table;
    DlRuleConfig config;
    
public:
    RuleManagerImpl() {
        memset(&builtin_best64, 0, sizeof(builtin_best64));
        memset(&function_table, 0, sizeof(function_table));
        
        // Default configuration
        config.enable_ptx_rules = true;
        config.enable_rule_chaining = true;
        config.max_chain_length = 4;
        config.enable_rule_caching = true;
        config.max_cache_size_mb = 64;
        config.enable_rule_analysis = true;
        config.prefer_speed_over_coverage = false;
    }
    
    bool load_builtin_rules() {
        builtin_best64.rule_count = 64;
        builtin_best64.rules = (DlCompiledRule*)malloc(64 * sizeof(DlCompiledRule));
        strcpy(builtin_best64.name, "Best64");
        strcpy(builtin_best64.description, "Most effective 64 password transformation rules");
        builtin_best64.is_builtin = true;
        
        // Compile built-in rules
        for (uint32_t i = 0; i < 64; ++i) {
            DlCompiledRule& rule = builtin_best64.rules[i];
            
            rule.type = RULE_TYPE_BUILTIN_PTX;
            rule.rule_id = i;
            strcpy(rule.rule_string, DL_BEST64_RULES[i]);
            
            // Set up parameters based on rule type
            compile_rule_parameters(DL_BEST64_RULES[i], &rule.params);
            rule.estimated_cost = estimate_rule_cost_internal(DL_BEST64_RULES[i]);
            rule.success_rate_permille = 50; // Default estimate
        }
        
        builtin_best64.total_combinations = calculate_total_combinations(&builtin_best64);
        return true;
    }
    
    bool load_ptx_rules() {
        // Initialize PTX function table
        for (uint32_t i = 0; i < 64; ++i) {
            function_table.ptx_functions[i] = ptx_best64_functions[i];
            function_table.ptx_available[i] = (ptx_best64_functions[i] != nullptr);
        }
        return true;
    }
    
    bool load_user_rules_from_file(const char* filepath, const char* name) {
        std::ifstream file(filepath);
        if (!file.is_open()) return false;
        
        std::vector<std::string> rule_strings;
        std::string line;
        
        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#') continue;
            
            // Validate rule
            if (validate_rule_string(line.c_str())) {
                rule_strings.push_back(line);
            }
        }
        
        file.close();
        
        if (rule_strings.empty()) return false;
        
        // Create user rule set
        DlRuleSet user_set;
        user_set.rule_count = rule_strings.size();
        user_set.rules = (DlCompiledRule*)malloc(user_set.rule_count * sizeof(DlCompiledRule));
        strcpy(user_set.name, name);
        strcpy(user_set.description, "User-uploaded rule set");
        user_set.is_builtin = false;
        
        // Compile user rules
        for (size_t i = 0; i < rule_strings.size(); ++i) {
            DlCompiledRule& rule = user_set.rules[i];
            
            rule.type = RULE_TYPE_USER_INTERPRETED;
            rule.rule_id = i;
            strcpy(rule.rule_string, rule_strings[i].c_str());
            
            compile_rule_parameters(rule_strings[i].c_str(), &rule.params);
            rule.estimated_cost = estimate_rule_cost_internal(rule_strings[i].c_str());
            rule.success_rate_permille = 25; // Conservative estimate for user rules
        }
        
        user_set.total_combinations = calculate_total_combinations(&user_set);
        user_rule_sets.push_back(user_set);
        
        return true;
    }
    
    bool validate_rule_string(const char* rule) {
        if (!rule || strlen(rule) == 0 || strlen(rule) > DL_MAX_RULE_LENGTH) {
            return false;
        }
        
        // Basic rule validation
        const char* r = rule;
        
        if (strcmp(r, ":") == 0) return true;  // no-op
        if (strcmp(r, "l") == 0) return true;  // lowercase
        if (strcmp(r, "u") == 0) return true;  // uppercase
        if (strcmp(r, "c") == 0) return true;  // capitalize
        if (strcmp(r, "C") == 0) return true;  // invert case
        if (strcmp(r, "t") == 0) return true;  // toggle case
        if (strcmp(r, "r") == 0) return true;  // reverse
        if (strcmp(r, "d") == 0) return true;  // duplicate
        if (strcmp(r, "f") == 0) return true;  // reflect
        if (strcmp(r, "[") == 0) return true;  // delete first
        if (strcmp(r, "]") == 0) return true;  // delete last
        
        // Append/prepend single character
        if (strlen(r) == 2 && (r[0] == '$' || r[0] == '^')) {
            return true;
        }
        
        // Character substitution
        if (strlen(r) == 3 && r[0] == 's') {
            return true;
        }
        
        // Delete at position
        if (strlen(r) == 2 && r[0] == 'D' && r[1] >= '0' && r[1] <= '9') {
            return true;
        }
        
        // More complex rules can be added here
        return false;
    }
    
    void execute_rule_batch_gpu(const DlRuleSet* rule_set,
                               const uint8_t* input_words, const uint32_t* input_lengths,
                               uint8_t* output_candidates, uint32_t* output_lengths,
                               uint32_t word_count, uint32_t max_output_length) {
        
        // Allocate GPU memory
        uint8_t* d_input_words;
        uint32_t* d_input_lengths;
        DlCompiledRule* d_rules;
        uint8_t* d_output_candidates;
        uint32_t* d_output_lengths;
        DlTelemetry* d_telemetry;
        
        size_t input_words_size = word_count * max_output_length;
        size_t output_size = word_count * rule_set->rule_count * max_output_length;
        
        cudaMalloc(&d_input_words, input_words_size);
        cudaMalloc(&d_input_lengths, word_count * sizeof(uint32_t));
        cudaMalloc(&d_rules, rule_set->rule_count * sizeof(DlCompiledRule));
        cudaMalloc(&d_output_candidates, output_size);
        cudaMalloc(&d_output_lengths, word_count * rule_set->rule_count * sizeof(uint32_t));
        cudaMalloc(&d_telemetry, sizeof(DlTelemetry));
        
        // Copy data to GPU
        cudaMemcpy(d_input_words, input_words, input_words_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_input_lengths, input_lengths, word_count * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rules, rule_set->rules, rule_set->rule_count * sizeof(DlCompiledRule), cudaMemcpyHostToDevice);
        cudaMemset(d_telemetry, 0, sizeof(DlTelemetry));
        
        // Launch kernel
        int device;
        cudaGetDevice(&device);
        KernelOptParams params = get_optimal_launch_params(device);
        
        hybrid_rule_execution_kernel<<<params.blocks, params.threads_per_block>>>(
            d_input_words, d_input_lengths, d_rules,
            d_output_candidates, d_output_lengths,
            word_count, rule_set->rule_count, max_output_length, d_telemetry);
        
        cudaDeviceSynchronize();
        
        // Copy results back
        cudaMemcpy(output_candidates, d_output_candidates, output_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(output_lengths, d_output_lengths, 
                   word_count * rule_set->rule_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        
        // Cleanup
        cudaFree(d_input_words);
        cudaFree(d_input_lengths);
        cudaFree(d_rules);
        cudaFree(d_output_candidates);
        cudaFree(d_output_lengths);
        cudaFree(d_telemetry);
    }
    
private:
    void compile_rule_parameters(const char* rule_str, DlRuleParams* params) {
        memset(params, 0, sizeof(DlRuleParams));
        
        params->variant_count = 1;
        params->max_length_delta = 0;
        
        if (rule_str[0] == '$' || rule_str[0] == '^') {
            // Append/prepend adds 1 character
            params->max_length_delta = 1;
            params->param_count = 1;
            params->params[0] = rule_str[1];
        }
        else if (rule_str[0] == 's' && strlen(rule_str) == 3) {
            // Substitution
            params->param_count = 2;
            params->params[0] = rule_str[1]; // from char
            params->params[1] = rule_str[2]; // to char
        }
        else if (strcmp(rule_str, "d") == 0) {
            // Duplicate doubles length
            params->max_length_delta = 256; // Assume max input length
        }
        else if (strcmp(rule_str, "[") == 0 || strcmp(rule_str, "]") == 0) {
            // Delete reduces length by 1
            params->max_length_delta = -1;
        }
    }
    
    float estimate_rule_cost_internal(const char* rule_str) {
        // Simple cost estimation based on rule complexity
        if (strcmp(rule_str, ":") == 0) return 0.1f;  // no-op is free
        if (rule_str[0] == '$' || rule_str[0] == '^') return 0.2f;  // append/prepend
        if (strcmp(rule_str, "l") == 0 || strcmp(rule_str, "u") == 0) return 0.3f;  // case change
        if (strcmp(rule_str, "r") == 0) return 0.5f;  // reverse
        if (strcmp(rule_str, "d") == 0) return 0.8f;  // duplicate
        return 0.4f;  // default
    }
    
    uint64_t calculate_total_combinations(const DlRuleSet* rule_set) {
        uint64_t total = 0;
        for (uint32_t i = 0; i < rule_set->rule_count; ++i) {
            total += rule_set->rules[i].params.variant_count;
        }
        return total;
    }
};

// C interface implementation
extern "C" {
    
DlRuleManager* dl_create_rule_manager(void) {
    return reinterpret_cast<DlRuleManager*>(new RuleManagerImpl());
}

void dl_destroy_rule_manager(DlRuleManager* manager) {
    delete reinterpret_cast<RuleManagerImpl*>(manager);
}

bool dl_load_builtin_rules(DlRuleManager* manager) {
    return reinterpret_cast<RuleManagerImpl*>(manager)->load_builtin_rules();
}

bool dl_load_ptx_rules(DlRuleManager* manager) {
    return reinterpret_cast<RuleManagerImpl*>(manager)->load_ptx_rules();
}

const DlRuleSet* dl_get_builtin_best64(DlRuleManager* manager) {
    if (!manager) return nullptr;
    RuleManagerImpl* impl = reinterpret_cast<RuleManagerImpl*>(manager);
    return &impl->builtin_best64;
}

bool dl_load_user_rules_from_file(DlRuleManager* manager, const char* filepath, const char* name) {
    return reinterpret_cast<RuleManagerImpl*>(manager)->load_user_rules_from_file(filepath, name);
}

bool dl_validate_rule_string(const char* rule) {
    RuleManagerImpl temp;
    return temp.validate_rule_string(rule);
}

void dl_execute_rule_batch_gpu(const DlRuleSet* rule_set,
                               const uint8_t* input_words, const uint32_t* input_lengths,
                               uint8_t* output_candidates, uint32_t* output_lengths,
                               uint32_t word_count, uint32_t max_output_length) {
    RuleManagerImpl temp;
    temp.execute_rule_batch_gpu(rule_set, input_words, input_lengths,
                               output_candidates, output_lengths, word_count, max_output_length);
}

const char* dl_rule_error_string(enum DlRuleError error) {
    switch (error) {
        case DL_RULE_SUCCESS: return "Success";
        case DL_RULE_ERROR_INVALID_SYNTAX: return "Invalid rule syntax";
        case DL_RULE_ERROR_TOO_LONG: return "Rule string too long";
        case DL_RULE_ERROR_UNSUPPORTED: return "Unsupported rule type";
        case DL_RULE_ERROR_MEMORY: return "Memory allocation error";
        case DL_RULE_ERROR_FILE_IO: return "File I/O error";
        case DL_RULE_ERROR_PTX_NOT_AVAILABLE: return "PTX implementation not available";
        default: return "Unknown error";
    }
}

} // extern "C"