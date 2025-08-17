#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <cmath>
#include "darkling_mask_engine.h"
#include "darkling_rules.h"
#include "hash_primitives.cuh"

// Smart rule ordering and optimization system
// Analyzes rule effectiveness and reorders for maximum efficiency

struct RuleEffectiveness {
    uint32_t rule_id;
    uint32_t variant_id;
    float success_rate;      // Hits per candidates generated
    uint64_t total_tested;   // Total candidates tested with this rule
    uint64_t total_hits;     // Total hits with this rule
    float avg_execution_time_us; // Average execution time
    float efficiency_score;  // Composite efficiency metric
};

struct RulePattern {
    uint32_t rule_sequence[8];  // Up to 8 rules in sequence
    uint32_t sequence_length;
    float pattern_success_rate;
    uint32_t pattern_frequency; // How often this pattern succeeds
};

// GPU kernel for rule effectiveness analysis
__global__ void analyze_rule_effectiveness_kernel(
    const DlDictWord* dict_words,
    const uint8_t* dict_data,
    uint32_t* rule_hit_counts,     // [num_rules * max_variants]
    uint32_t* rule_test_counts,    // [num_rules * max_variants]
    float* rule_exec_times,        // [num_rules * max_variants]
    uint32_t dict_word_count,
    uint32_t num_rules,
    uint32_t max_variants,
    uint32_t batch_size) {
    
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= batch_size) return;
    
    // Sample dictionary words for testing
    uint32_t word_index = tid % dict_word_count;
    DlDictWord word = dict_words[word_index];
    
    // Test each rule with multiple variants
    for (uint32_t rule_id = 0; rule_id < num_rules; ++rule_id) {
        DlRuleMC rule = g_rules[rule_id];
        DlRuleDispatch disp = g_dispatch[rule.shape];
        
        for (uint32_t variant = 0; variant < min(max_variants, disp.variants); ++variant) {
            uint32_t index = rule_id * max_variants + variant;
            
            // Apply rule transformation
            uint8_t temp_buffer[128];
            uint8_t candidate[128];
            const uint8_t* word_data = dict_data + word.offset;
            
            // Copy base word
            for (uint32_t i = 0; i < word.length; ++i) {
                temp_buffer[i] = word_data[i];
            }
            
            // Record execution time
            clock_t start_time = clock();
            
            RuleParams params;
            for (int j = 0; j < 16; ++j) {
                params.bytes[j] = rule.params[j];
            }
            
            disp.fn(candidate, temp_buffer, word.length, &params, variant, disp.variants);
            
            clock_t end_time = clock();
            float exec_time = (float)(end_time - start_time);
            
            // Update execution time statistics
            atomicAdd((unsigned long long*)&rule_exec_times[index], (unsigned long long)(exec_time * 1000000));
            
            // Hash and check for effectiveness (simplified)
            uint32_t final_length = word.length + rule.length_delta;
            uint32_t hash[4];
            md5_hash(candidate, final_length, hash);
            
            // Increment test count
            atomicAdd(&rule_test_counts[index], 1);
            
            // Simple effectiveness check (in real implementation, check against target)
            if (check_hash(hash)) {
                atomicAdd(&rule_hit_counts[index], 1);
            }
        }
    }
}

// Host function for smart rule optimization
class SmartRuleOptimizer {
private:
    std::vector<RuleEffectiveness> rule_stats;
    std::vector<RulePattern> successful_patterns;
    std::unordered_map<uint32_t, float> rule_complexity_scores;
    
public:
    SmartRuleOptimizer() {
        // Initialize complexity scores for different rule types
        rule_complexity_scores[PREFIX_1] = 0.1f;      // Very simple
        rule_complexity_scores[SUFFIX_D4] = 0.2f;     // Simple
        rule_complexity_scores[CASE_TOGGLE] = 0.3f;   // Moderate
        rule_complexity_scores[LEET_LIGHT] = 0.5f;    // Complex
        rule_complexity_scores[SUFFIX_SHORT] = 0.2f;  // Simple
        rule_complexity_scores[AFFIX_PAIR] = 0.7f;    // Very complex
    }
    
    void analyze_rule_performance(
        const DlDictionary* dictionary,
        uint32_t sample_size = 10000) {
        
        // Allocate GPU memory for analysis
        uint32_t* d_hit_counts;
        uint32_t* d_test_counts;
        float* d_exec_times;
        
        const uint32_t num_rules = 256;
        const uint32_t max_variants = 100;
        
        cudaMalloc(&d_hit_counts, num_rules * max_variants * sizeof(uint32_t));
        cudaMalloc(&d_test_counts, num_rules * max_variants * sizeof(uint32_t));
        cudaMalloc(&d_exec_times, num_rules * max_variants * sizeof(float));
        
        cudaMemset(d_hit_counts, 0, num_rules * max_variants * sizeof(uint32_t));
        cudaMemset(d_test_counts, 0, num_rules * max_variants * sizeof(uint32_t));
        cudaMemset(d_exec_times, 0, num_rules * max_variants * sizeof(float));
        
        // Upload dictionary to GPU
        DlDictWord* d_dict_words;
        uint8_t* d_dict_data;
        cudaMalloc(&d_dict_words, dictionary->word_count * sizeof(DlDictWord));
        cudaMalloc(&d_dict_data, dictionary->data_size);
        cudaMemcpy(d_dict_words, dictionary->words, dictionary->word_count * sizeof(DlDictWord), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dict_data, dictionary->data, dictionary->data_size, cudaMemcpyHostToDevice);
        
        // Launch analysis kernel
        dim3 blocks((sample_size + 255) / 256);
        dim3 threads(256);
        
        analyze_rule_effectiveness_kernel<<<blocks, threads>>>(
            d_dict_words, d_dict_data, d_hit_counts, d_test_counts, d_exec_times,
            dictionary->word_count, num_rules, max_variants, sample_size);
        
        cudaDeviceSynchronize();
        
        // Download results
        std::vector<uint32_t> hit_counts(num_rules * max_variants);
        std::vector<uint32_t> test_counts(num_rules * max_variants);
        std::vector<float> exec_times(num_rules * max_variants);
        
        cudaMemcpy(hit_counts.data(), d_hit_counts, hit_counts.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(test_counts.data(), d_test_counts, test_counts.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(exec_times.data(), d_exec_times, exec_times.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Analyze results and build effectiveness statistics
        rule_stats.clear();
        for (uint32_t rule_id = 0; rule_id < num_rules; ++rule_id) {
            for (uint32_t variant = 0; variant < max_variants; ++variant) {
                uint32_t index = rule_id * max_variants + variant;
                
                if (test_counts[index] > 0) {
                    RuleEffectiveness eff;
                    eff.rule_id = rule_id;
                    eff.variant_id = variant;
                    eff.total_tested = test_counts[index];
                    eff.total_hits = hit_counts[index];
                    eff.success_rate = (float)hit_counts[index] / test_counts[index];
                    eff.avg_execution_time_us = exec_times[index] / test_counts[index];
                    
                    // Calculate composite efficiency score
                    float complexity = rule_complexity_scores[rule_id % 6]; // Map to known rule types
                    eff.efficiency_score = eff.success_rate / (complexity * eff.avg_execution_time_us + 1e-6f);
                    
                    rule_stats.push_back(eff);
                }
            }
        }
        
        // Sort by efficiency score
        std::sort(rule_stats.begin(), rule_stats.end(), 
                  [](const RuleEffectiveness& a, const RuleEffectiveness& b) {
                      return a.efficiency_score > b.efficiency_score;
                  });
        
        // Cleanup
        cudaFree(d_hit_counts);
        cudaFree(d_test_counts);
        cudaFree(d_exec_times);
        cudaFree(d_dict_words);
        cudaFree(d_dict_data);
    }
    
    DlRuleChain generate_optimal_rule_chain(uint32_t max_chain_length = 4) {
        DlRuleChain chain = {};
        
        if (rule_stats.empty()) {
            // Default chain if no analysis data
            chain.rule_count = 2;
            chain.rules[0] = 0; // PREFIX_1
            chain.rules[1] = 1; // SUFFIX_D4
            chain.variant_counts[0] = 14;
            chain.variant_counts[1] = 10000;
            chain.max_variants = 10000;
            return chain;
        }
        
        // Select top performing rules
        uint32_t selected_rules = 0;
        for (const auto& eff : rule_stats) {
            if (selected_rules >= max_chain_length) break;
            
            // Avoid duplicate rule types in chain
            bool duplicate = false;
            for (uint32_t i = 0; i < selected_rules; ++i) {
                if (chain.rules[i] == eff.rule_id) {
                    duplicate = true;
                    break;
                }
            }
            
            if (!duplicate && eff.efficiency_score > 0.01f) {
                chain.rules[selected_rules] = eff.rule_id;
                
                // Limit variants based on effectiveness
                uint32_t optimal_variants = std::min(
                    (uint32_t)(eff.success_rate * 1000 + 1),
                    (uint32_t)100
                );
                chain.variant_counts[selected_rules] = optimal_variants;
                selected_rules++;
            }
        }
        
        chain.rule_count = selected_rules;
        
        // Calculate max variants
        chain.max_variants = 1;
        for (uint32_t i = 0; i < chain.rule_count; ++i) {
            chain.max_variants *= chain.variant_counts[i];
        }
        
        return chain;
    }
    
    void identify_successful_patterns() {
        // Machine learning approach to identify successful rule patterns
        std::unordered_map<std::string, RulePattern> pattern_map;
        
        // Analyze combinations of 2-4 rules
        for (size_t i = 0; i < rule_stats.size() && i < 20; ++i) {
            for (size_t j = i + 1; j < rule_stats.size() && j < 20; ++j) {
                // Two-rule pattern
                std::string pattern_key = std::to_string(rule_stats[i].rule_id) + 
                                        "_" + std::to_string(rule_stats[j].rule_id);
                
                RulePattern pattern;
                pattern.rule_sequence[0] = rule_stats[i].rule_id;
                pattern.rule_sequence[1] = rule_stats[j].rule_id;
                pattern.sequence_length = 2;
                pattern.pattern_success_rate = (rule_stats[i].success_rate + rule_stats[j].success_rate) / 2.0f;
                pattern.pattern_frequency = 1;
                
                pattern_map[pattern_key] = pattern;
                
                // Three-rule patterns
                for (size_t k = j + 1; k < rule_stats.size() && k < 15; ++k) {
                    std::string pattern3_key = pattern_key + "_" + std::to_string(rule_stats[k].rule_id);
                    
                    RulePattern pattern3;
                    pattern3.rule_sequence[0] = rule_stats[i].rule_id;
                    pattern3.rule_sequence[1] = rule_stats[j].rule_id;
                    pattern3.rule_sequence[2] = rule_stats[k].rule_id;
                    pattern3.sequence_length = 3;
                    pattern3.pattern_success_rate = (rule_stats[i].success_rate + 
                                                   rule_stats[j].success_rate + 
                                                   rule_stats[k].success_rate) / 3.0f;
                    pattern3.pattern_frequency = 1;
                    
                    pattern_map[pattern3_key] = pattern3;
                }
            }
        }
        
        // Convert to vector and sort by success rate
        successful_patterns.clear();
        for (const auto& pair : pattern_map) {
            successful_patterns.push_back(pair.second);
        }
        
        std::sort(successful_patterns.begin(), successful_patterns.end(),
                  [](const RulePattern& a, const RulePattern& b) {
                      return a.pattern_success_rate > b.pattern_success_rate;
                  });
    }
    
    std::vector<DlRuleChain> generate_adaptive_rule_chains(uint32_t num_chains = 5) {
        std::vector<DlRuleChain> chains;
        
        // Generate diverse rule chains based on different strategies
        
        // 1. High-frequency rules chain
        DlRuleChain freq_chain = generate_optimal_rule_chain(3);
        chains.push_back(freq_chain);
        
        // 2. Low-complexity high-speed chain
        DlRuleChain speed_chain = {};
        speed_chain.rule_count = 2;
        speed_chain.rules[0] = 0; // PREFIX_1 (simple)
        speed_chain.rules[1] = 4; // SUFFIX_SHORT (simple)
        speed_chain.variant_counts[0] = 10;
        speed_chain.variant_counts[1] = 100;
        speed_chain.max_variants = 1000;
        chains.push_back(speed_chain);
        
        // 3. Complex transformation chain
        DlRuleChain complex_chain = {};
        complex_chain.rule_count = 4;
        complex_chain.rules[0] = 2; // CASE_TOGGLE
        complex_chain.rules[1] = 3; // LEET_LIGHT
        complex_chain.rules[2] = 1; // SUFFIX_D4
        complex_chain.rules[3] = 5; // AFFIX_PAIR
        complex_chain.variant_counts[0] = 5;
        complex_chain.variant_counts[1] = 10;
        complex_chain.variant_counts[2] = 50;
        complex_chain.variant_counts[3] = 10;
        complex_chain.max_variants = 25000;
        chains.push_back(complex_chain);
        
        // 4. Pattern-based chain (if patterns identified)
        if (!successful_patterns.empty()) {
            DlRuleChain pattern_chain = {};
            const RulePattern& best_pattern = successful_patterns[0];
            pattern_chain.rule_count = best_pattern.sequence_length;
            
            for (uint32_t i = 0; i < best_pattern.sequence_length; ++i) {
                pattern_chain.rules[i] = best_pattern.rule_sequence[i];
                pattern_chain.variant_counts[i] = 20; // Moderate variants
            }
            
            pattern_chain.max_variants = 1;
            for (uint32_t i = 0; i < pattern_chain.rule_count; ++i) {
                pattern_chain.max_variants *= pattern_chain.variant_counts[i];
            }
            
            chains.push_back(pattern_chain);
        }
        
        // 5. Balanced effectiveness chain
        DlRuleChain balanced_chain = generate_optimal_rule_chain(3);
        // Adjust variants for balance
        for (uint32_t i = 0; i < balanced_chain.rule_count; ++i) {
            balanced_chain.variant_counts[i] = std::min(balanced_chain.variant_counts[i], 50u);
        }
        chains.push_back(balanced_chain);
        
        return chains;
    }
    
    void print_analysis_results() {
        printf("Smart Rule Analysis Results:\n");
        printf("===========================\n");
        
        printf("Top 10 Most Efficient Rules:\n");
        for (size_t i = 0; i < std::min(rule_stats.size(), size_t(10)); ++i) {
            const auto& eff = rule_stats[i];
            printf("Rule %u Variant %u: Success=%.4f%% Efficiency=%.6f Time=%.2fμs\n",
                   eff.rule_id, eff.variant_id, eff.success_rate * 100,
                   eff.efficiency_score, eff.avg_execution_time_us);
        }
        
        printf("\nSuccessful Rule Patterns:\n");
        for (size_t i = 0; i < std::min(successful_patterns.size(), size_t(5)); ++i) {
            const auto& pattern = successful_patterns[i];
            printf("Pattern %zu: Rules [", i);
            for (uint32_t j = 0; j < pattern.sequence_length; ++j) {
                printf("%u%s", pattern.rule_sequence[j], 
                       j < pattern.sequence_length - 1 ? "," : "");
            }
            printf("] Success=%.4f%%\n", pattern.pattern_success_rate * 100);
        }
    }
};

// C interface for integration
extern "C" {
    SmartRuleOptimizer* dl_create_rule_optimizer() {
        return new SmartRuleOptimizer();
    }
    
    void dl_destroy_rule_optimizer(SmartRuleOptimizer* optimizer) {
        delete optimizer;
    }
    
    void dl_analyze_rule_performance(SmartRuleOptimizer* optimizer, const DlDictionary* dict) {
        optimizer->analyze_rule_performance(dict);
    }
    
    DlRuleChain dl_generate_optimal_chain(SmartRuleOptimizer* optimizer, uint32_t max_length) {
        return optimizer->generate_optimal_rule_chain(max_length);
    }
}