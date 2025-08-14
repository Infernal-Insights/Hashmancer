#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include "darkling_mask_engine.h"
#include "darkling_rules.h"
#include "hash_primitives.cuh"
#include "kernel_optimizations.cuh"

// Hybrid attack modes: dictionary+mask, mask+dictionary, dictionary+rules combinations

// GPU kernel for dictionary + mask hybrid attack
__global__ void hybrid_dict_mask_attack_kernel(
    const DlDictWord* dict_words,
    const uint8_t* dict_data,
    uint8_t* candidates,
    uint32_t* lengths,
    uint32_t* hash_outputs,
    uint32_t dict_word_count,
    uint64_t mask_keyspace,
    uint64_t global_offset,
    uint32_t batch_size,
    uint32_t max_length,
    DlTelemetry* telemetry) {
    
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    
    for (uint32_t i = tid; i < batch_size; i += stride) {
        uint64_t candidate_index = global_offset + i;
        
        // Calculate which dictionary word and mask combination
        uint32_t dict_index = candidate_index / mask_keyspace;
        uint64_t mask_index = candidate_index % mask_keyspace;
        
        if (dict_index >= dict_word_count) continue;
        
        DlDictWord word = dict_words[dict_index];
        uint8_t* candidate = candidates + i * max_length;
        uint32_t pos = 0;
        
        // Copy dictionary word first
        const uint8_t* word_data = dict_data + word.offset;
        for (uint32_t w = 0; w < word.length && pos < max_length - g_mask_length; ++w) {
            candidate[pos++] = word_data[w];
        }
        
        // Generate mask portion
        uint64_t remaining_mask = mask_index;
        for (uint32_t p = 0; p < g_mask_length && pos < max_length; ++p) {
            DlMaskPos mask_pos = g_mask_positions[p];
            
            if (mask_pos.type == MASK_LITERAL) {
                candidate[pos++] = g_mask_charsets[mask_pos.start_offset];
            } else {
                uint32_t char_index = remaining_mask % mask_pos.char_count;
                remaining_mask /= mask_pos.char_count;
                candidate[pos++] = g_mask_charsets[mask_pos.start_offset + char_index];
            }
        }
        
        lengths[i] = pos;
        
        // Hash the candidate
        uint32_t* hash_out = hash_outputs + i * 4;
        md5_hash(candidate, pos, hash_out);
        
        // Check for hit
        if (check_hash(hash_out)) {
            atomicAdd(&telemetry->hits, 1ULL);
        }
        
        atomicAdd(&telemetry->candidates_generated, 1ULL);
    }
}

// GPU kernel for mask + dictionary hybrid attack
__global__ void hybrid_mask_dict_attack_kernel(
    const DlDictWord* dict_words,
    const uint8_t* dict_data,
    uint8_t* candidates,
    uint32_t* lengths,
    uint32_t* hash_outputs,
    uint32_t dict_word_count,
    uint64_t mask_keyspace,
    uint64_t global_offset,
    uint32_t batch_size,
    uint32_t max_length,
    DlTelemetry* telemetry) {
    
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    
    for (uint32_t i = tid; i < batch_size; i += stride) {
        uint64_t candidate_index = global_offset + i;
        
        // Calculate which mask and dictionary word combination
        uint32_t dict_index = candidate_index / mask_keyspace;
        uint64_t mask_index = candidate_index % mask_keyspace;
        
        if (dict_index >= dict_word_count) continue;
        
        DlDictWord word = dict_words[dict_index];
        uint8_t* candidate = candidates + i * max_length;
        uint32_t pos = 0;
        
        // Generate mask portion first
        uint64_t remaining_mask = mask_index;
        for (uint32_t p = 0; p < g_mask_length && pos < max_length; ++p) {
            DlMaskPos mask_pos = g_mask_positions[p];
            
            if (mask_pos.type == MASK_LITERAL) {
                candidate[pos++] = g_mask_charsets[mask_pos.start_offset];
            } else {
                uint32_t char_index = remaining_mask % mask_pos.char_count;
                remaining_mask /= mask_pos.char_count;
                candidate[pos++] = g_mask_charsets[mask_pos.start_offset + char_index];
            }
        }
        
        // Append dictionary word
        const uint8_t* word_data = dict_data + word.offset;
        for (uint32_t w = 0; w < word.length && pos < max_length; ++w) {
            candidate[pos++] = word_data[w];
        }
        
        lengths[i] = pos;
        
        // Hash the candidate
        uint32_t* hash_out = hash_outputs + i * 4;
        md5_hash(candidate, pos, hash_out);
        
        // Check for hit
        if (check_hash(hash_out)) {
            atomicAdd(&telemetry->hits, 1ULL);
        }
        
        atomicAdd(&telemetry->candidates_generated, 1ULL);
    }
}

// Advanced rule chaining kernel
__global__ void dict_rules_chain_kernel(
    const DlDictWord* dict_words,
    const uint8_t* dict_data,
    const DlRuleChain* rule_chain,
    uint8_t* candidates,
    uint32_t* lengths,
    uint32_t* hash_outputs,
    uint32_t dict_word_count,
    uint64_t rule_combinations,
    uint64_t global_offset,
    uint32_t batch_size,
    uint32_t max_length,
    DlTelemetry* telemetry) {
    
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    
    // Shared memory for intermediate transformations
    extern __shared__ uint8_t shared_buffers[];
    uint8_t* temp_buffer = shared_buffers + threadIdx.x * max_length;
    
    for (uint32_t i = tid; i < batch_size; i += stride) {
        uint64_t candidate_index = global_offset + i;
        
        // Decode dictionary word and rule combination
        uint32_t dict_index = candidate_index / rule_combinations;
        uint64_t rule_combination = candidate_index % rule_combinations;
        
        if (dict_index >= dict_word_count) continue;
        
        DlDictWord word = dict_words[dict_index];
        uint8_t* candidate = candidates + i * max_length;
        
        // Start with base dictionary word
        const uint8_t* word_data = dict_data + word.offset;
        uint32_t current_length = word.length;
        
        // Copy to working buffer
        for (uint32_t w = 0; w < current_length; ++w) {
            temp_buffer[w] = word_data[w];
        }
        
        // Apply rule chain
        uint64_t remaining_combination = rule_combination;
        uint8_t* input_buffer = temp_buffer;
        uint8_t* output_buffer = candidate;
        
        for (uint32_t r = 0; r < rule_chain->rule_count; ++r) {
            uint32_t rule_index = rule_chain->rules[r];
            uint32_t variant_count = rule_chain->variant_counts[r];
            uint32_t variant_index = remaining_combination % variant_count;
            remaining_combination /= variant_count;
            
            // Apply rule transformation
            DlRuleMC rule = g_rules[rule_index];
            DlRuleDispatch disp = g_dispatch[rule.shape];
            RuleParams params;
            
            for (int j = 0; j < 16; ++j) {
                params.bytes[j] = rule.params[j];
            }
            
            // Transform input to output
            disp.fn(output_buffer, input_buffer, current_length, &params, variant_index, variant_count);
            current_length += rule.length_delta;
            
            // Swap buffers for next iteration
            uint8_t* tmp = input_buffer;
            input_buffer = output_buffer;
            output_buffer = tmp;
        }
        
        // Ensure final result is in candidate buffer
        if (input_buffer != candidate) {
            for (uint32_t c = 0; c < current_length; ++c) {
                candidate[c] = input_buffer[c];
            }
        }
        
        lengths[i] = current_length;
        
        // Hash the final candidate
        uint32_t* hash_out = hash_outputs + i * 4;
        md5_hash(candidate, current_length, hash_out);
        
        // Check for hit
        if (check_hash(hash_out)) {
            atomicAdd(&telemetry->hits, 1ULL);
        }
        
        atomicAdd(&telemetry->candidates_generated, 1ULL);
    }
}

// Host function to launch hybrid attack
extern "C" void dl_hybrid_attack_launch(
    DlAttackMode mode,
    const DlDictionary* dictionary,
    const DlMask* mask,
    const DlRuleChain* rule_chain,
    uint8_t* d_candidates,
    uint32_t* d_lengths,
    uint32_t* d_hash_outputs,
    uint64_t global_offset,
    uint32_t batch_size,
    uint32_t max_length,
    DlTelemetry* d_telemetry) {
    
    // Get optimal launch parameters
    int device;
    cudaGetDevice(&device);
    KernelOptParams params = get_optimal_launch_params(device);
    
    // Calculate shared memory requirements
    uint32_t shared_mem_per_thread = max_length * 2; // For temp buffers
    uint32_t total_shared_mem = shared_mem_per_thread * params.threads_per_block;
    
    switch (mode) {
        case ATTACK_HYBRID_DICT_MASK: {
            uint64_t mask_keyspace = dl_mask_calculate_keyspace(mask);
            hybrid_dict_mask_attack_kernel<<<params.blocks, params.threads_per_block>>>(
                dictionary->words, dictionary->data, d_candidates, d_lengths, d_hash_outputs,
                dictionary->word_count, mask_keyspace, global_offset, batch_size, max_length, d_telemetry);
            break;
        }
        
        case ATTACK_HYBRID_MASK_DICT: {
            uint64_t mask_keyspace = dl_mask_calculate_keyspace(mask);
            hybrid_mask_dict_attack_kernel<<<params.blocks, params.threads_per_block>>>(
                dictionary->words, dictionary->data, d_candidates, d_lengths, d_hash_outputs,
                dictionary->word_count, mask_keyspace, global_offset, batch_size, max_length, d_telemetry);
            break;
        }
        
        case ATTACK_DICT_RULES: {
            // Calculate total rule combinations
            uint64_t rule_combinations = 1;
            for (uint32_t r = 0; r < rule_chain->rule_count; ++r) {
                rule_combinations *= rule_chain->variant_counts[r];
            }
            
            // Limit shared memory usage
            total_shared_mem = min(total_shared_mem, 48*1024u); // Max 48KB
            
            dict_rules_chain_kernel<<<params.blocks, params.threads_per_block, total_shared_mem>>>(
                dictionary->words, dictionary->data, rule_chain, d_candidates, d_lengths, d_hash_outputs,
                dictionary->word_count, rule_combinations, global_offset, batch_size, max_length, d_telemetry);
            break;
        }
        
        default:
            // Fallback to basic mask attack
            dl_mask_launch_kernel(nullptr, nullptr, d_candidates, d_lengths, global_offset, batch_size, max_length);
            break;
    }
}

// Dictionary preprocessing and optimization
extern "C" bool dl_dictionary_preprocess(
    const char* wordlist_path,
    DlDictionary* output_dict,
    uint32_t min_length,
    uint32_t max_length,
    bool sort_by_frequency,
    bool deduplicate) {
    
    FILE* file = fopen(wordlist_path, "rb");
    if (!file) return false;
    
    // Get file size
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Read entire file
    char* file_data = (char*)malloc(file_size + 1);
    fread(file_data, 1, file_size, file);
    file_data[file_size] = '\0';
    fclose(file);
    
    // Parse words
    std::vector<std::string> words;
    std::unordered_map<std::string, uint32_t> word_frequencies;
    
    char* line = strtok(file_data, "\n\r");
    while (line) {
        uint32_t len = strlen(line);
        if (len >= min_length && len <= max_length) {
            std::string word(line);
            words.push_back(word);
            
            if (sort_by_frequency) {
                word_frequencies[word]++;
            }
        }
        line = strtok(nullptr, "\n\r");
    }
    
    free(file_data);
    
    // Deduplicate if requested
    if (deduplicate) {
        std::sort(words.begin(), words.end());
        words.erase(std::unique(words.begin(), words.end()), words.end());
    }
    
    // Sort by frequency if requested
    if (sort_by_frequency) {
        std::sort(words.begin(), words.end(), [&](const std::string& a, const std::string& b) {
            return word_frequencies[a] > word_frequencies[b];
        });
    }
    
    // Build dictionary structure
    output_dict->word_count = words.size();
    output_dict->words = (DlDictWord*)malloc(sizeof(DlDictWord) * words.size());
    
    // Calculate total data size
    size_t total_data_size = 0;
    for (const auto& word : words) {
        total_data_size += word.length();
    }
    
    output_dict->data_size = total_data_size;
    output_dict->data = (uint8_t*)malloc(total_data_size);
    
    // Copy word data
    size_t data_offset = 0;
    for (size_t i = 0; i < words.size(); ++i) {
        output_dict->words[i].offset = data_offset;
        output_dict->words[i].length = words[i].length();
        output_dict->words[i].frequency = sort_by_frequency ? word_frequencies[words[i]] : 1;
        
        memcpy(output_dict->data + data_offset, words[i].c_str(), words[i].length());
        data_offset += words[i].length();
    }
    
    return true;
}

// Calculate keyspace for hybrid attacks
uint64_t dl_hybrid_calculate_keyspace(const DlMask* mask, const DlDictionary* dict) {
    uint64_t mask_keyspace = dl_mask_calculate_keyspace(mask);
    uint64_t dict_size = dict->word_count;
    return mask_keyspace * dict_size;
}