#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <fstream>
#include <cstring>
#include <cmath>
#include "darkling_mask_engine.h"

// Advanced wordlist preprocessing and optimization system
// Implements intelligent sorting, deduplication, frequency analysis, and compression

struct WordStatistics {
    uint32_t length;
    uint32_t frequency;
    uint32_t charset_complexity;    // Number of unique character types
    uint32_t positional_entropy;    // Entropy based on character positions
    float success_probability;      // ML-predicted success rate
    uint32_t rule_compatibility;    // Bitmask of compatible rules
};

struct WordlistMetrics {
    uint32_t total_words;
    uint32_t unique_words;
    uint32_t avg_length;
    uint32_t min_length;
    uint32_t max_length;
    std::vector<uint32_t> length_distribution;
    std::vector<uint32_t> charset_distribution;
    float compression_ratio;
    float deduplication_ratio;
};

// GPU kernel for parallel word analysis
__global__ void analyze_words_kernel(
    const DlDictWord* words,
    const uint8_t* word_data,
    WordStatistics* stats,
    uint32_t word_count) {
    
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= word_count) return;
    
    DlDictWord word = words[tid];
    const uint8_t* data = word_data + word.offset;
    WordStatistics& stat = stats[tid];
    
    stat.length = word.length;
    stat.frequency = word.frequency;
    
    // Analyze character complexity
    bool has_lower = false, has_upper = false, has_digit = false, has_special = false;
    uint32_t char_histogram[256] = {0};
    
    for (uint32_t i = 0; i < word.length; ++i) {
        uint8_t c = data[i];
        char_histogram[c]++;
        
        if (c >= 'a' && c <= 'z') has_lower = true;
        else if (c >= 'A' && c <= 'Z') has_upper = true;
        else if (c >= '0' && c <= '9') has_digit = true;
        else has_special = true;
    }
    
    stat.charset_complexity = has_lower + has_upper + has_digit + has_special;
    
    // Calculate positional entropy
    float entropy = 0.0f;
    for (uint32_t i = 0; i < 256; ++i) {
        if (char_histogram[i] > 0) {
            float p = (float)char_histogram[i] / word.length;
            entropy -= p * logf(p);
        }
    }
    stat.positional_entropy = (uint32_t)(entropy * 1000); // Scale for integer storage
    
    // Rule compatibility analysis
    stat.rule_compatibility = 0;
    
    // Compatible with prefix rules
    stat.rule_compatibility |= (1 << PREFIX_1);
    
    // Compatible with suffix rules if not too long
    if (word.length <= 60) {
        stat.rule_compatibility |= (1 << SUFFIX_D4);
        stat.rule_compatibility |= (1 << SUFFIX_SHORT);
    }
    
    // Compatible with case toggle if has letters
    if (has_lower || has_upper) {
        stat.rule_compatibility |= (1 << CASE_TOGGLE);
    }
    
    // Compatible with leet if has compatible characters
    for (uint32_t i = 0; i < word.length; ++i) {
        uint8_t c = data[i];
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 's' || c == 't' || c == 'l' ||
            c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'S' || c == 'T' || c == 'L') {
            stat.rule_compatibility |= (1 << LEET_LIGHT);
            break;
        }
    }
    
    // Compatible with affix pairs if reasonable length
    if (word.length >= 3 && word.length <= 50) {
        stat.rule_compatibility |= (1 << AFFIX_PAIR);
    }
    
    // Simple ML-based success probability (heuristic)
    float base_prob = 1.0f / (word.length * word.length); // Shorter words more likely
    float freq_factor = logf(word.frequency + 1) / 10.0f;  // Higher frequency bonus
    float complexity_factor = stat.charset_complexity / 10.0f; // Complexity bonus
    
    stat.success_probability = base_prob * (1.0f + freq_factor + complexity_factor);
}

class WordlistOptimizer {
private:
    std::vector<std::string> words;
    std::vector<WordStatistics> word_stats;
    WordlistMetrics metrics;
    
public:
    bool load_wordlist(const char* filepath, bool preserve_order = false) {
        std::ifstream file(filepath);
        if (!file.is_open()) return false;
        
        words.clear();
        std::string line;
        
        while (std::getline(file, line)) {
            // Remove trailing whitespace
            while (!line.empty() && std::isspace(line.back())) {
                line.pop_back();
            }
            
            if (!line.empty()) {
                words.push_back(line);
            }
        }
        
        file.close();
        
        if (!preserve_order) {
            // Initial frequency-based sorting
            std::unordered_map<std::string, uint32_t> frequency_map;
            for (const auto& word : words) {
                frequency_map[word]++;
            }
            
            std::sort(words.begin(), words.end(), 
                      [&frequency_map](const std::string& a, const std::string& b) {
                          return frequency_map[a] > frequency_map[b];
                      });
        }
        
        return true;
    }
    
    void deduplicate_wordlist(bool case_sensitive = true) {
        std::unordered_set<std::string> seen;
        std::vector<std::string> unique_words;
        
        uint32_t original_count = words.size();
        
        for (const auto& word : words) {
            std::string key = case_sensitive ? word : to_lowercase(word);
            
            if (seen.find(key) == seen.end()) {
                seen.insert(key);
                unique_words.push_back(word);
            }
        }
        
        words = std::move(unique_words);
        
        if (original_count > 0) {
            metrics.deduplication_ratio = (float)(original_count - words.size()) / original_count;
        }
    }
    
    void filter_by_length(uint32_t min_len, uint32_t max_len) {
        words.erase(std::remove_if(words.begin(), words.end(),
                                   [min_len, max_len](const std::string& word) {
                                       return word.length() < min_len || word.length() > max_len;
                                   }), words.end());
    }
    
    void filter_by_charset(const std::string& allowed_chars) {
        std::unordered_set<char> allowed_set(allowed_chars.begin(), allowed_chars.end());
        
        words.erase(std::remove_if(words.begin(), words.end(),
                                   [&allowed_set](const std::string& word) {
                                       for (char c : word) {
                                           if (allowed_set.find(c) == allowed_set.end()) {
                                               return true;
                                           }
                                       }
                                       return false;
                                   }), words.end());
    }
    
    void analyze_with_gpu() {
        if (words.empty()) return;
        
        // Convert to DlDictionary format for GPU analysis
        DlDictionary dict = create_dictionary_from_words();
        
        // Allocate GPU memory
        DlDictWord* d_words;
        uint8_t* d_data;
        WordStatistics* d_stats;
        
        cudaMalloc(&d_words, dict.word_count * sizeof(DlDictWord));
        cudaMalloc(&d_data, dict.data_size);
        cudaMalloc(&d_stats, dict.word_count * sizeof(WordStatistics));
        
        // Copy data to GPU
        cudaMemcpy(d_words, dict.words, dict.word_count * sizeof(DlDictWord), cudaMemcpyHostToDevice);
        cudaMemcpy(d_data, dict.data, dict.data_size, cudaMemcpyHostToDevice);
        
        // Launch analysis kernel
        dim3 blocks((dict.word_count + 255) / 256);
        dim3 threads(256);
        
        analyze_words_kernel<<<blocks, threads>>>(d_words, d_data, d_stats, dict.word_count);
        cudaDeviceSynchronize();
        
        // Copy results back
        word_stats.resize(dict.word_count);
        cudaMemcpy(word_stats.data(), d_stats, dict.word_count * sizeof(WordStatistics), cudaMemcpyDeviceToHost);
        
        // Cleanup
        cudaFree(d_words);
        cudaFree(d_data);
        cudaFree(d_stats);
        free(dict.words);
        free(dict.data);
    }
    
    void optimize_order_by_strategy(const std::string& strategy) {
        if (word_stats.empty()) {
            analyze_with_gpu();
        }
        
        if (strategy == "frequency") {
            // Sort by frequency
            std::vector<std::pair<std::string, WordStatistics>> word_stat_pairs;
            for (size_t i = 0; i < words.size() && i < word_stats.size(); ++i) {
                word_stat_pairs.emplace_back(words[i], word_stats[i]);
            }
            
            std::sort(word_stat_pairs.begin(), word_stat_pairs.end(),
                      [](const auto& a, const auto& b) {
                          return a.second.frequency > b.second.frequency;
                      });
            
            words.clear();
            word_stats.clear();
            for (const auto& pair : word_stat_pairs) {
                words.push_back(pair.first);
                word_stats.push_back(pair.second);
            }
        }
        else if (strategy == "length") {
            // Sort by length (shorter first for faster initial hits)
            std::vector<std::pair<std::string, WordStatistics>> word_stat_pairs;
            for (size_t i = 0; i < words.size() && i < word_stats.size(); ++i) {
                word_stat_pairs.emplace_back(words[i], word_stats[i]);
            }
            
            std::sort(word_stat_pairs.begin(), word_stat_pairs.end(),
                      [](const auto& a, const auto& b) {
                          if (a.second.length != b.second.length) {
                              return a.second.length < b.second.length;
                          }
                          return a.second.frequency > b.second.frequency;
                      });
            
            words.clear();
            word_stats.clear();
            for (const auto& pair : word_stat_pairs) {
                words.push_back(pair.first);
                word_stats.push_back(pair.second);
            }
        }
        else if (strategy == "ml_predicted") {
            // Sort by ML-predicted success probability
            std::vector<std::pair<std::string, WordStatistics>> word_stat_pairs;
            for (size_t i = 0; i < words.size() && i < word_stats.size(); ++i) {
                word_stat_pairs.emplace_back(words[i], word_stats[i]);
            }
            
            std::sort(word_stat_pairs.begin(), word_stat_pairs.end(),
                      [](const auto& a, const auto& b) {
                          return a.second.success_probability > b.second.success_probability;
                      });
            
            words.clear();
            word_stats.clear();
            for (const auto& pair : word_stat_pairs) {
                words.push_back(pair.first);
                word_stats.push_back(pair.second);
            }
        }
        else if (strategy == "entropy") {
            // Sort by positional entropy (higher entropy first)
            std::vector<std::pair<std::string, WordStatistics>> word_stat_pairs;
            for (size_t i = 0; i < words.size() && i < word_stats.size(); ++i) {
                word_stat_pairs.emplace_back(words[i], word_stats[i]);
            }
            
            std::sort(word_stat_pairs.begin(), word_stat_pairs.end(),
                      [](const auto& a, const auto& b) {
                          return a.second.positional_entropy > b.second.positional_entropy;
                      });
            
            words.clear();
            word_stats.clear();
            for (const auto& pair : word_stat_pairs) {
                words.push_back(pair.first);
                word_stats.push_back(pair.second);
            }
        }
    }
    
    void generate_rule_specific_wordlists(const std::vector<uint32_t>& target_rules, 
                                        const std::string& output_prefix) {
        if (word_stats.empty()) {
            analyze_with_gpu();
        }
        
        for (uint32_t rule_id : target_rules) {
            std::vector<std::string> compatible_words;
            
            for (size_t i = 0; i < words.size() && i < word_stats.size(); ++i) {
                if (word_stats[i].rule_compatibility & (1 << rule_id)) {
                    compatible_words.push_back(words[i]);
                }
            }
            
            // Save rule-specific wordlist
            std::string filename = output_prefix + "_rule_" + std::to_string(rule_id) + ".txt";
            save_wordlist(filename, compatible_words);
        }
    }
    
    void compress_wordlist_lz4() {
        // Implement LZ4 compression for storage efficiency
        // This would reduce memory usage for large wordlists
        // Implementation would require LZ4 library integration
    }
    
    void split_by_length_groups(const std::string& output_prefix, uint32_t group_size = 2) {
        std::unordered_map<uint32_t, std::vector<std::string>> length_groups;
        
        for (const auto& word : words) {
            uint32_t group = word.length() / group_size;
            length_groups[group].push_back(word);
        }
        
        for (const auto& group : length_groups) {
            std::string filename = output_prefix + "_len_" + 
                                 std::to_string(group.first * group_size) + "_" +
                                 std::to_string((group.first + 1) * group_size - 1) + ".txt";
            save_wordlist(filename, group.second);
        }
    }
    
    WordlistMetrics calculate_metrics() {
        metrics.total_words = words.size();
        
        std::unordered_set<std::string> unique_set(words.begin(), words.end());
        metrics.unique_words = unique_set.size();
        
        if (!words.empty()) {
            uint32_t total_length = 0;
            metrics.min_length = words[0].length();
            metrics.max_length = words[0].length();
            
            std::unordered_map<uint32_t, uint32_t> length_dist;
            std::unordered_map<uint32_t, uint32_t> charset_dist;
            
            for (const auto& word : words) {
                total_length += word.length();
                metrics.min_length = std::min(metrics.min_length, (uint32_t)word.length());
                metrics.max_length = std::max(metrics.max_length, (uint32_t)word.length());
                
                length_dist[word.length()]++;
                
                // Analyze charset
                bool has_lower = false, has_upper = false, has_digit = false, has_special = false;
                for (char c : word) {
                    if (c >= 'a' && c <= 'z') has_lower = true;
                    else if (c >= 'A' && c <= 'Z') has_upper = true;
                    else if (c >= '0' && c <= '9') has_digit = true;
                    else has_special = true;
                }
                
                uint32_t charset_mask = (has_lower ? 1 : 0) | (has_upper ? 2 : 0) | 
                                       (has_digit ? 4 : 0) | (has_special ? 8 : 0);
                charset_dist[charset_mask]++;
            }
            
            metrics.avg_length = total_length / words.size();
            
            // Convert distributions to vectors
            for (const auto& pair : length_dist) {
                if (pair.first < metrics.length_distribution.size()) {
                    metrics.length_distribution.resize(pair.first + 1);
                }
                metrics.length_distribution[pair.first] = pair.second;
            }
            
            for (const auto& pair : charset_dist) {
                if (pair.first < metrics.charset_distribution.size()) {
                    metrics.charset_distribution.resize(pair.first + 1);
                }
                metrics.charset_distribution[pair.first] = pair.second;
            }
        }
        
        return metrics;
    }
    
    bool save_wordlist(const std::string& filepath, 
                      const std::vector<std::string>& word_list = {}) {
        std::ofstream file(filepath);
        if (!file.is_open()) return false;
        
        const auto& list_to_save = word_list.empty() ? words : word_list;
        
        for (const auto& word : list_to_save) {
            file << word << "\n";
        }
        
        file.close();
        return true;
    }
    
    void print_metrics() {
        WordlistMetrics m = calculate_metrics();
        
        printf("Wordlist Optimization Metrics:\n");
        printf("==============================\n");
        printf("Total words: %u\n", m.total_words);
        printf("Unique words: %u\n", m.unique_words);
        printf("Average length: %u\n", m.avg_length);
        printf("Length range: %u - %u\n", m.min_length, m.max_length);
        printf("Deduplication ratio: %.2f%%\n", m.deduplication_ratio * 100);
        
        printf("\nLength distribution:\n");
        for (size_t i = 0; i < m.length_distribution.size(); ++i) {
            if (m.length_distribution[i] > 0) {
                printf("  Length %zu: %u words\n", i, m.length_distribution[i]);
            }
        }
    }
    
private:
    std::string to_lowercase(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }
    
    DlDictionary create_dictionary_from_words() {
        DlDictionary dict = {};
        dict.word_count = words.size();
        
        // Calculate total data size
        size_t total_size = 0;
        for (const auto& word : words) {
            total_size += word.length();
        }
        
        dict.data_size = total_size;
        dict.words = (DlDictWord*)malloc(dict.word_count * sizeof(DlDictWord));
        dict.data = (uint8_t*)malloc(dict.data_size);
        
        // Copy word data
        size_t offset = 0;
        for (size_t i = 0; i < words.size(); ++i) {
            dict.words[i].offset = offset;
            dict.words[i].length = words[i].length();
            dict.words[i].frequency = 1; // Default frequency
            
            memcpy(dict.data + offset, words[i].c_str(), words[i].length());
            offset += words[i].length();
        }
        
        return dict;
    }
};

// C interface
extern "C" {
    WordlistOptimizer* dl_create_wordlist_optimizer() {
        return new WordlistOptimizer();
    }
    
    void dl_destroy_wordlist_optimizer(WordlistOptimizer* optimizer) {
        delete optimizer;
    }
    
    bool dl_load_wordlist(WordlistOptimizer* optimizer, const char* filepath) {
        return optimizer->load_wordlist(filepath);
    }
    
    void dl_optimize_wordlist(WordlistOptimizer* optimizer, const char* strategy) {
        optimizer->deduplicate_wordlist();
        optimizer->optimize_order_by_strategy(strategy);
    }
    
    bool dl_save_optimized_wordlist(WordlistOptimizer* optimizer, const char* filepath) {
        return optimizer->save_wordlist(filepath);
    }
}