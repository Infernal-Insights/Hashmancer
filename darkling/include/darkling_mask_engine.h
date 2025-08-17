#pragma once
#include <stdint.h>
#include <stdbool.h>

// Mask engine for GPU-accelerated brute force attacks
// Supports hashcat-style mask syntax: ?l?d?s?u?a?b

// Mask character types
enum DlMaskType : uint8_t {
    MASK_LOWER = 'l',        // lowercase letters
    MASK_UPPER = 'u',        // uppercase letters  
    MASK_DIGIT = 'd',        // digits 0-9
    MASK_SPECIAL = 's',      // special characters !@#$%^&*
    MASK_ALL = 'a',          // all printable ASCII
    MASK_BINARY = 'b',       // all bytes 0x00-0xFF
    MASK_HEX_LOWER = 'h',    // hex digits 0-9a-f
    MASK_HEX_UPPER = 'H',    // hex digits 0-9A-F
    MASK_CUSTOM = 'c',       // custom charset from file
    MASK_LITERAL = 0xFF      // literal character
};

// Mask position descriptor
struct DlMaskPos {
    uint8_t type;           // DlMaskType
    uint8_t charset_id;     // For custom charsets
    uint8_t char_count;     // Number of characters in this position
    uint8_t padding;
    uint32_t start_offset;  // Offset into charset data
};

// Compiled mask for GPU execution
struct DlMask {
    uint32_t length;                    // Mask length in positions
    uint32_t max_candidates;            // Total combinations
    uint64_t charset_data_size;         // Size of charset data buffer
    DlMaskPos positions[32];            // Position descriptors (max 32 chars)
    uint8_t* charset_data;              // Character data for all positions
};

// Mask compiler result
struct DlMaskCompileResult {
    bool success;
    DlMask mask;
    char error_msg[256];
};

// Host functions for mask compilation
DlMaskCompileResult dl_mask_compile(const char* mask_string);
void dl_mask_destroy(DlMask* mask);
bool dl_mask_upload_to_gpu(const DlMask* mask, void** d_mask, void** d_charset_data);
void dl_mask_free_gpu(void* d_mask, void* d_charset_data);

// GPU kernel for mask-based candidate generation
void dl_mask_launch_kernel(
    const void* d_mask,
    const void* d_charset_data,
    uint8_t* d_candidates,      // Output buffer [batch_size * max_length]
    uint32_t* d_lengths,        // Output lengths [batch_size]
    uint64_t start_index,       // Starting candidate index
    uint32_t batch_size,        // Number of candidates to generate
    uint32_t max_length         // Maximum candidate length
);

// Hybrid attack structures
struct DlHybridConfig {
    bool dict_first;            // true: dict+mask, false: mask+dict
    uint32_t dict_min_len;      // Minimum dictionary word length
    uint32_t dict_max_len;      // Maximum dictionary word length
    uint32_t mask_min_len;      // Minimum mask length
    uint32_t mask_max_len;      // Maximum mask length
    uint32_t total_max_len;     // Maximum combined length
};

// Dictionary word descriptor
struct DlDictWord {
    uint32_t offset;            // Offset in dictionary data
    uint16_t length;            // Word length
    uint16_t frequency;         // Word frequency (for prioritization)
};

// Dictionary structure optimized for GPU access
struct DlDictionary {
    uint32_t word_count;        // Number of words
    uint32_t data_size;         // Size of word data buffer
    DlDictWord* words;          // Word descriptors
    uint8_t* data;              // Word data buffer
    uint32_t* frequency_sorted; // Word indices sorted by frequency
};

// Rule chain for advanced transformations
struct DlRuleChain {
    uint32_t rule_count;        // Number of rules in chain
    uint32_t max_variants;      // Maximum variants per rule
    uint32_t rules[16];         // Rule indices (max 16 rules per chain)
    uint32_t variant_counts[16]; // Variants per rule
};

// Attack mode configuration
enum DlAttackMode {
    ATTACK_MASK_ONLY = 0,       // Pure mask attack
    ATTACK_DICT_ONLY = 1,       // Pure dictionary attack
    ATTACK_DICT_RULES = 2,      // Dictionary + rules
    ATTACK_HYBRID_DICT_MASK = 3, // Dictionary + mask hybrid
    ATTACK_HYBRID_MASK_DICT = 4, // Mask + dictionary hybrid
    ATTACK_MARKOV_CHAIN = 5,    // Markov chain attack
    ATTACK_PRINCE = 6,          // PRINCE algorithm
    ATTACK_COMBINATOR = 7       // Combinator attack
};

// Main attack configuration
struct DlAttackConfig {
    DlAttackMode mode;
    DlMask mask;
    DlDictionary dictionary;
    DlRuleChain rule_chain;
    DlHybridConfig hybrid_config;
    uint64_t skip_count;        // Skip first N candidates
    uint64_t limit_count;       // Process only N candidates (0 = unlimited)
    uint32_t gpu_batch_size;    // Candidates per GPU batch
    bool resume_from_checkpoint; // Resume from saved position
};

// Performance monitoring for mask attacks
struct DlMaskStats {
    uint64_t candidates_tested;
    uint64_t candidates_per_second;
    uint64_t time_elapsed_ms;
    uint64_t estimated_total_time_ms;
    uint32_t gpu_utilization;
    uint32_t memory_utilization;
    double completion_percentage;
};

// Advanced mask optimization
struct DlMaskOptimizer {
    bool enable_position_reordering;  // Reorder positions by charset size
    bool enable_early_rejection;      // Skip impossible combinations
    bool enable_smart_skip;           // Skip based on previous results
    uint32_t checkpoint_interval;     // Save state every N candidates
    uint32_t max_memory_mb;          // Maximum GPU memory usage
};

// Mask keyspace calculation
uint64_t dl_mask_calculate_keyspace(const DlMask* mask);
uint64_t dl_hybrid_calculate_keyspace(const DlMask* mask, const DlDictionary* dict);
bool dl_mask_estimate_runtime(const DlMask* mask, uint64_t hash_rate, uint64_t* estimated_seconds);

// Checkpoint/resume functionality
bool dl_mask_save_checkpoint(const char* filename, uint64_t position, const DlAttackConfig* config);
bool dl_mask_load_checkpoint(const char* filename, uint64_t* position, DlAttackConfig* config);

// Multi-GPU coordination
struct DlMaskDistribution {
    uint32_t gpu_count;
    uint32_t gpu_id;            // Current GPU ID
    uint64_t total_keyspace;
    uint64_t gpu_start_offset;   // Starting position for this GPU
    uint64_t gpu_end_offset;     // Ending position for this GPU
};

bool dl_mask_distribute_keyspace(const DlMask* mask, uint32_t gpu_count, DlMaskDistribution* distribution);

// Memory optimization for large keyspaces
struct DlMaskMemoryPool {
    void* gpu_candidate_buffer;
    void* gpu_length_buffer;
    void* gpu_hash_buffer;
    size_t buffer_size;
    uint32_t batch_count;
};

DlMaskMemoryPool* dl_mask_create_memory_pool(uint32_t max_batch_size, uint32_t max_length);
void dl_mask_destroy_memory_pool(DlMaskMemoryPool* pool);