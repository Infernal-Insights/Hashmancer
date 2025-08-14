#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <random>
#include "darkling_mask_engine.h"
#include "hash_primitives.cuh"
#include "kernel_optimizations.cuh"

// Markov Chain password generation engine
// Implements statistical modeling of password patterns for intelligent candidate generation

struct MarkovNode {
    uint8_t character;
    uint32_t frequency;
    float probability;
    uint32_t next_count;
    uint32_t next_offset;   // Offset into transitions array
};

struct MarkovTransition {
    uint8_t to_char;
    uint32_t frequency;
    float probability;
    float cumulative_prob;
};

struct MarkovModel {
    uint32_t order;         // N-gram order (1=unigram, 2=bigram, 3=trigram, etc.)
    uint32_t node_count;    // Total number of nodes
    uint32_t transition_count; // Total number of transitions
    MarkovNode* nodes;      // Character nodes
    MarkovTransition* transitions; // Character transitions
    uint32_t* char_to_node; // Lookup table [256] -> node index
    uint8_t start_chars[256]; // Possible starting characters
    uint32_t start_char_count;
    float* start_probabilities; // Starting character probabilities
};

// GPU constant memory for Markov model
__device__ __constant__ MarkovNode g_markov_nodes[8192];
__device__ __constant__ MarkovTransition g_markov_transitions[32768];
__device__ __constant__ uint32_t g_markov_char_to_node[256];
__device__ __constant__ uint8_t g_markov_start_chars[256];
__device__ __constant__ float g_markov_start_probs[256];
__device__ __constant__ uint32_t g_markov_start_count;
__device__ __constant__ uint32_t g_markov_order;

// GPU kernel for Markov chain password generation
__global__ void markov_generate_kernel(
    uint8_t* candidates,        // [batch_size * max_length]
    uint32_t* lengths,          // [batch_size]
    uint64_t* rng_states,       // [batch_size] RNG state per thread
    uint32_t batch_size,
    uint32_t min_length,
    uint32_t max_length,
    float length_bias) {        // Bias towards certain lengths
    
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    
    for (uint32_t i = tid; i < batch_size; i += stride) {
        uint64_t rng_state = rng_states[i];
        uint8_t* candidate = candidates + i * max_length;
        uint32_t pos = 0;
        
        // Choose target length based on bias
        float length_rand = curand_uniform(&rng_state);
        uint32_t target_length;
        
        if (length_bias > 0.5f) {
            // Bias towards shorter passwords
            target_length = min_length + (uint32_t)(powf(length_rand, 2.0f) * (max_length - min_length));
        } else {
            // Uniform distribution
            target_length = min_length + (uint32_t)(length_rand * (max_length - min_length));
        }
        
        // Choose starting character
        float start_rand = curand_uniform(&rng_state);
        uint8_t current_char = g_markov_start_chars[0]; // Default fallback
        
        for (uint32_t s = 0; s < g_markov_start_count; ++s) {
            if (start_rand <= g_markov_start_probs[s]) {
                current_char = g_markov_start_chars[s];
                break;
            }
        }
        
        candidate[pos++] = current_char;
        
        // Generate subsequent characters using Markov chain
        uint8_t context[8] = {0}; // Support up to 8-gram
        uint32_t context_len = min(g_markov_order, 8u);
        context[0] = current_char;
        
        while (pos < target_length && pos < max_length) {
            // Build context key for current position
            uint32_t context_hash = 0;
            for (uint32_t c = 0; c < context_len && c < pos; ++c) {
                context_hash = context_hash * 256 + context[c];
            }
            
            // Find node for current context
            uint32_t node_index = g_markov_char_to_node[current_char];
            if (node_index >= 8192) {
                // Fallback: choose random character
                current_char = 'a' + (curand(&rng_state) % 26);
                candidate[pos++] = current_char;
                continue;
            }
            
            MarkovNode node = g_markov_nodes[node_index];
            
            // Choose next character based on transition probabilities
            float transition_rand = curand_uniform(&rng_state);
            bool found_transition = false;
            
            for (uint32_t t = 0; t < node.next_count; ++t) {
                uint32_t trans_index = node.next_offset + t;
                if (trans_index >= 32768) break;
                
                MarkovTransition trans = g_markov_transitions[trans_index];
                if (transition_rand <= trans.cumulative_prob) {
                    current_char = trans.to_char;
                    found_transition = true;
                    break;
                }
            }
            
            if (!found_transition) {
                // Fallback: choose from common characters
                const char common_chars[] = "aeiounrtlsdhgmfpbwyckvjxqz";
                current_char = common_chars[curand(&rng_state) % 26];
            }
            
            candidate[pos++] = current_char;
            
            // Update context
            for (uint32_t c = context_len - 1; c > 0; --c) {
                context[c] = context[c - 1];
            }
            context[0] = current_char;
        }
        
        lengths[i] = pos;
        rng_states[i] = rng_state; // Save RNG state
    }
}

// Advanced Markov chain with position-aware modeling
__global__ void markov_positional_generate_kernel(
    uint8_t* candidates,
    uint32_t* lengths,
    uint64_t* rng_states,
    uint32_t batch_size,
    uint32_t min_length,
    uint32_t max_length) {
    
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= batch_size) return;
    
    uint64_t rng_state = rng_states[tid];
    uint8_t* candidate = candidates + tid * max_length;
    
    // Generate length based on observed length distribution
    float length_rand = curand_uniform(&rng_state);
    uint32_t target_length = min_length + (uint32_t)(length_rand * (max_length - min_length));
    
    for (uint32_t pos = 0; pos < target_length && pos < max_length; ++pos) {
        uint8_t next_char;
        
        if (pos == 0) {
            // Starting character from start distribution
            float start_rand = curand_uniform(&rng_state);
            next_char = g_markov_start_chars[0];
            
            for (uint32_t s = 0; s < g_markov_start_count; ++s) {
                if (start_rand <= g_markov_start_probs[s]) {
                    next_char = g_markov_start_chars[s];
                    break;
                }
            }
        } else {
            // Use previous characters as context
            uint8_t prev_char = candidate[pos - 1];
            uint32_t node_index = g_markov_char_to_node[prev_char];
            
            if (node_index < 8192) {
                MarkovNode node = g_markov_nodes[node_index];
                float trans_rand = curand_uniform(&rng_state);
                
                next_char = 'a'; // Default
                for (uint32_t t = 0; t < node.next_count && t < 32; ++t) {
                    uint32_t trans_index = node.next_offset + t;
                    if (trans_index < 32768) {
                        MarkovTransition trans = g_markov_transitions[trans_index];
                        if (trans_rand <= trans.cumulative_prob) {
                            next_char = trans.to_char;
                            break;
                        }
                    }
                }
            } else {
                // Fallback to position-based character selection
                if (pos < target_length / 2) {
                    // First half: favor vowels and common consonants
                    const char first_half[] = "aeiourtnlsdhgmfp";
                    next_char = first_half[curand(&rng_state) % 16];
                } else {
                    // Second half: favor consonants and numbers
                    const char second_half[] = "rtnlsdhgmfpbwyckvjxqz123456789";
                    next_char = second_half[curand(&rng_state) % 29];
                }
            }
        }
        
        candidate[pos] = next_char;
    }
    
    lengths[tid] = target_length;
    rng_states[tid] = rng_state;
}

// Host implementation
class MarkovChainEngine {
private:
    MarkovModel model;
    std::unordered_map<std::string, uint32_t> ngram_counts;
    std::unordered_map<uint8_t, uint32_t> char_counts;
    uint32_t total_chars;
    
public:
    MarkovChainEngine(uint32_t order = 2) {
        model.order = order;
        model.node_count = 0;
        model.transition_count = 0;
        model.nodes = nullptr;
        model.transitions = nullptr;
        model.char_to_node = nullptr;
        model.start_probabilities = nullptr;
        model.start_char_count = 0;
        total_chars = 0;
    }
    
    ~MarkovChainEngine() {
        cleanup();
    }
    
    void cleanup() {
        if (model.nodes) free(model.nodes);
        if (model.transitions) free(model.transitions);
        if (model.char_to_node) free(model.char_to_node);
        if (model.start_probabilities) free(model.start_probabilities);
        
        model.nodes = nullptr;
        model.transitions = nullptr;
        model.char_to_node = nullptr;
        model.start_probabilities = nullptr;
    }
    
    bool train_from_wordlist(const char* wordlist_path) {
        FILE* file = fopen(wordlist_path, "r");
        if (!file) return false;
        
        char line[256];
        while (fgets(line, sizeof(line), file)) {
            // Remove newline
            size_t len = strlen(line);
            if (len > 0 && line[len - 1] == '\n') {
                line[len - 1] = '\0';
                len--;
            }
            
            if (len > 0) {
                train_from_password(line, len);
            }
        }
        
        fclose(file);
        return build_model();
    }
    
    void train_from_password(const char* password, size_t length) {
        // Count character frequencies
        for (size_t i = 0; i < length; ++i) {
            char_counts[(uint8_t)password[i]]++;
            total_chars++;
        }
        
        // Count n-grams
        for (size_t i = 0; i <= length - model.order; ++i) {
            std::string ngram(password + i, model.order);
            ngram_counts[ngram]++;
        }
        
        // Count starting characters
        if (length > 0) {
            char_counts[(uint8_t)password[0]]++;
        }
    }
    
    bool build_model() {
        if (char_counts.empty()) return false;
        
        // Build character to node mapping
        model.char_to_node = (uint32_t*)malloc(256 * sizeof(uint32_t));
        std::fill(model.char_to_node, model.char_to_node + 256, UINT32_MAX);
        
        // Create nodes for each character
        std::vector<MarkovNode> nodes;
        std::vector<MarkovTransition> transitions;
        
        for (const auto& char_pair : char_counts) {
            uint8_t c = char_pair.first;
            uint32_t freq = char_pair.second;
            
            MarkovNode node;
            node.character = c;
            node.frequency = freq;
            node.probability = (float)freq / total_chars;
            node.next_count = 0;
            node.next_offset = transitions.size();
            
            model.char_to_node[c] = nodes.size();
            
            // Find all transitions from this character
            std::unordered_map<uint8_t, uint32_t> next_char_counts;
            uint32_t total_next = 0;
            
            for (const auto& ngram_pair : ngram_counts) {
                const std::string& ngram = ngram_pair.first;
                uint32_t count = ngram_pair.second;
                
                if (ngram.length() >= model.order && ngram[0] == c) {
                    uint8_t next_char = (uint8_t)ngram[1];
                    next_char_counts[next_char] += count;
                    total_next += count;
                }
            }
            
            // Create transitions
            float cumulative = 0.0f;
            for (const auto& next_pair : next_char_counts) {
                uint8_t next_char = next_pair.first;
                uint32_t count = next_pair.second;
                
                MarkovTransition trans;
                trans.to_char = next_char;
                trans.frequency = count;
                trans.probability = (float)count / total_next;
                cumulative += trans.probability;
                trans.cumulative_prob = cumulative;
                
                transitions.push_back(trans);
                node.next_count++;
            }
            
            nodes.push_back(node);
        }
        
        // Build starting character distribution
        std::vector<uint8_t> start_chars;
        std::vector<float> start_probs;
        
        for (const auto& char_pair : char_counts) {
            start_chars.push_back(char_pair.first);
            start_probs.push_back((float)char_pair.second / total_chars);
        }
        
        // Sort by probability (descending)
        std::vector<size_t> indices(start_chars.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return start_probs[a] > start_probs[b];
        });
        
        // Rearrange and calculate cumulative probabilities
        std::vector<uint8_t> sorted_chars;
        std::vector<float> cumulative_probs;
        float cumulative = 0.0f;
        
        for (size_t idx : indices) {
            sorted_chars.push_back(start_chars[idx]);
            cumulative += start_probs[idx];
            cumulative_probs.push_back(cumulative);
        }
        
        // Allocate and copy model data
        model.node_count = nodes.size();
        model.transition_count = transitions.size();
        model.start_char_count = std::min((size_t)256, sorted_chars.size());
        
        model.nodes = (MarkovNode*)malloc(model.node_count * sizeof(MarkovNode));
        model.transitions = (MarkovTransition*)malloc(model.transition_count * sizeof(MarkovTransition));
        model.start_probabilities = (float*)malloc(model.start_char_count * sizeof(float));
        
        memcpy(model.nodes, nodes.data(), model.node_count * sizeof(MarkovNode));
        memcpy(model.transitions, transitions.data(), model.transition_count * sizeof(MarkovTransition));
        
        for (uint32_t i = 0; i < model.start_char_count; ++i) {
            model.start_chars[i] = sorted_chars[i];
            model.start_probabilities[i] = cumulative_probs[i];
        }
        
        return true;
    }
    
    bool upload_model_to_gpu() {
        if (!model.nodes || !model.transitions) return false;
        
        // Upload to constant memory
        cudaError_t err;
        
        err = cudaMemcpyToSymbol(g_markov_nodes, model.nodes, 
                                std::min(model.node_count, 8192u) * sizeof(MarkovNode));
        if (err != cudaSuccess) return false;
        
        err = cudaMemcpyToSymbol(g_markov_transitions, model.transitions,
                                std::min(model.transition_count, 32768u) * sizeof(MarkovTransition));
        if (err != cudaSuccess) return false;
        
        err = cudaMemcpyToSymbol(g_markov_char_to_node, model.char_to_node, 256 * sizeof(uint32_t));
        if (err != cudaSuccess) return false;
        
        err = cudaMemcpyToSymbol(g_markov_start_chars, model.start_chars, 
                                std::min(model.start_char_count, 256u) * sizeof(uint8_t));
        if (err != cudaSuccess) return false;
        
        err = cudaMemcpyToSymbol(g_markov_start_probs, model.start_probabilities,
                                std::min(model.start_char_count, 256u) * sizeof(float));
        if (err != cudaSuccess) return false;
        
        err = cudaMemcpyToSymbol(g_markov_start_count, &model.start_char_count, sizeof(uint32_t));
        if (err != cudaSuccess) return false;
        
        err = cudaMemcpyToSymbol(g_markov_order, &model.order, sizeof(uint32_t));
        if (err != cudaSuccess) return false;
        
        return true;
    }
    
    void launch_generation(
        uint8_t* d_candidates,
        uint32_t* d_lengths,
        uint64_t* d_rng_states,
        uint32_t batch_size,
        uint32_t min_length,
        uint32_t max_length,
        bool use_positional_model = false) {
        
        // Get optimal launch parameters
        int device;
        cudaGetDevice(&device);
        KernelOptParams params = get_optimal_launch_params(device);
        
        if (use_positional_model) {
            markov_positional_generate_kernel<<<params.blocks, params.threads_per_block>>>(
                d_candidates, d_lengths, d_rng_states, batch_size, min_length, max_length);
        } else {
            markov_generate_kernel<<<params.blocks, params.threads_per_block>>>(
                d_candidates, d_lengths, d_rng_states, batch_size, min_length, max_length, 0.7f);
        }
    }
    
    void print_model_stats() {
        printf("Markov Chain Model Statistics:\n");
        printf("==============================\n");
        printf("Order: %u\n", model.order);
        printf("Nodes: %u\n", model.node_count);
        printf("Transitions: %u\n", model.transition_count);
        printf("Start characters: %u\n", model.start_char_count);
        printf("Total characters processed: %u\n", total_chars);
        
        printf("\nTop 10 most frequent characters:\n");
        std::vector<std::pair<uint32_t, uint8_t>> sorted_chars;
        for (const auto& pair : char_counts) {
            sorted_chars.emplace_back(pair.second, pair.first);
        }
        std::sort(sorted_chars.rbegin(), sorted_chars.rend());
        
        for (size_t i = 0; i < std::min(sorted_chars.size(), size_t(10)); ++i) {
            char c = sorted_chars[i].second;
            uint32_t freq = sorted_chars[i].first;
            printf("  '%c': %u (%.2f%%)\n", 
                   isprint(c) ? c : '?', freq, 100.0f * freq / total_chars);
        }
    }
};

// C interface
extern "C" {
    MarkovChainEngine* dl_create_markov_engine(uint32_t order) {
        return new MarkovChainEngine(order);
    }
    
    void dl_destroy_markov_engine(MarkovChainEngine* engine) {
        delete engine;
    }
    
    bool dl_train_markov_from_wordlist(MarkovChainEngine* engine, const char* wordlist_path) {
        return engine->train_from_wordlist(wordlist_path);
    }
    
    bool dl_upload_markov_model(MarkovChainEngine* engine) {
        return engine->upload_model_to_gpu();
    }
    
    void dl_markov_generate_candidates(
        MarkovChainEngine* engine,
        uint8_t* d_candidates,
        uint32_t* d_lengths,
        uint64_t* d_rng_states,
        uint32_t batch_size,
        uint32_t min_length,
        uint32_t max_length) {
        
        engine->launch_generation(d_candidates, d_lengths, d_rng_states, 
                                 batch_size, min_length, max_length);
    }
}