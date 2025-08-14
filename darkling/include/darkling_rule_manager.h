#pragma once
#include <stdint.h>
#include <stdbool.h>

// Advanced rule management system supporting both built-in PTX rules and user-uploaded rules
// Built-in rules use optimized PTX for maximum performance
// User rules use interpreted CUDA C++ for flexibility

// Rule types
enum DlRuleType {
    RULE_TYPE_BUILTIN_PTX = 0,    // Pre-compiled PTX rule (fastest)
    RULE_TYPE_USER_INTERPRETED = 1, // Runtime interpreted rule (flexible)
    RULE_TYPE_COMPILED_CUDA = 2    // User rule compiled to CUDA (future)
};

// Built-in rule identifiers (Best64 + extensions)
enum DlBuiltinRule {
    // Core Best64 rules (most effective 64 rules)
    RULE_BEST64_NOOP = 0,              // : (no operation)
    RULE_BEST64_LOWERCASE = 1,         // l (lowercase all)
    RULE_BEST64_UPPERCASE = 2,         // u (uppercase all)
    RULE_BEST64_CAPITALIZE = 3,        // c (capitalize first)
    RULE_BEST64_INVERT_CASE = 4,       // C (invert case)
    RULE_BEST64_TOGGLE_CASE = 5,       // t (toggle case)
    RULE_BEST64_REVERSE = 6,           // r (reverse)
    RULE_BEST64_DUPLICATE = 7,         // d (duplicate)
    RULE_BEST64_REFLECT = 8,           // f (reflect)
    RULE_BEST64_ROTATE_LEFT = 9,       // { (rotate left)
    RULE_BEST64_ROTATE_RIGHT = 10,     // } (rotate right)
    RULE_BEST64_DELETE_FIRST = 11,     // [ (delete first)
    RULE_BEST64_DELETE_LAST = 12,      // ] (delete last)
    RULE_BEST64_DELETE_AT_N = 13,      // DN (delete at position N)
    RULE_BEST64_EXTRACT_RANGE = 14,    // xNM (extract range)
    RULE_BEST64_INSERT_AT_N = 15,      // iNX (insert X at position N)
    RULE_BEST64_OVERWRITE_AT_N = 16,   // oNX (overwrite at position N)
    RULE_BEST64_TRUNCATE_AT_N = 17,    // 'N (truncate at position N)
    RULE_BEST64_APPEND_CHAR = 18,      // $X (append character X)
    RULE_BEST64_PREPEND_CHAR = 19,     // ^X (prepend character X)
    RULE_BEST64_REPLACE_CHAR = 20,     // sXY (replace X with Y)
    RULE_BEST64_PURGE_CHAR = 21,       // @X (purge character X)
    RULE_BEST64_DUPLICATE_FIRST = 22,  // p1 (duplicate first char)
    RULE_BEST64_DUPLICATE_LAST = 23,   // p2 (duplicate last char)
    RULE_BEST64_DUPLICATE_N = 24,      // pN (duplicate char at N)
    RULE_BEST64_APPEND_DIGIT = 25,     // $0-$9 (append digits)
    RULE_BEST64_PREPEND_DIGIT = 26,    // ^0-^9 (prepend digits)
    RULE_BEST64_APPEND_SYMBOL = 27,    // $! $@ $# etc (append symbols)
    RULE_BEST64_PREPEND_SYMBOL = 28,   // ^! ^@ ^# etc (prepend symbols)
    RULE_BEST64_LEET_COMMON = 29,      // Common leet substitutions
    RULE_BEST64_YEAR_APPEND = 30,      // Append years 00-99, 1900-2030
    RULE_BEST64_DOUBLE_WORD = 31,      // Double the word
    RULE_BEST64_TITLE_CASE = 32,       // Title case (first letter caps)
    
    // Extended rules (beyond Best64)
    RULE_EXT_ROT13 = 64,               // ROT13 encoding
    RULE_EXT_VOWEL_REPLACE = 65,       // Replace vowels with numbers
    RULE_EXT_CONSONANT_DOUBLE = 66,    // Double consonants
    RULE_EXT_KEYBOARD_SHIFT = 67,      // Keyboard shift patterns
    RULE_EXT_MORSE_CODE = 68,          // Morse code transformation
    RULE_EXT_PHONETIC = 69,            // Phonetic replacements
    
    RULE_BUILTIN_COUNT = 70            // Total built-in rules
};

// Rule parameter structure
struct DlRuleParams {
    uint8_t param_count;               // Number of parameters
    uint8_t params[16];                // Rule parameters
    uint32_t variant_count;            // Number of variants this rule generates
    uint32_t max_length_delta;         // Maximum length change (+/-)
};

// Compiled rule descriptor
struct DlCompiledRule {
    DlRuleType type;                   // Rule implementation type
    uint32_t rule_id;                  // Rule identifier
    DlRuleParams params;               // Rule parameters
    char rule_string[64];              // Original rule string (for debugging)
    float estimated_cost;              // Computational cost estimate
    uint32_t success_rate_permille;    // Success rate per 1000 (from analysis)
};

// Rule set container
struct DlRuleSet {
    uint32_t rule_count;               // Number of rules in set
    DlCompiledRule* rules;             // Array of compiled rules
    char name[64];                     // Rule set name
    char description[256];             // Rule set description
    bool is_builtin;                   // True if built-in rule set
    uint64_t total_combinations;       // Total rule combinations
};

// Rule manager context
struct DlRuleManager {
    DlRuleSet builtin_best64;          // Built-in Best64 rule set
    DlRuleSet* user_rule_sets;         // User-uploaded rule sets
    uint32_t user_rule_set_count;      // Number of user rule sets
    uint32_t max_user_rule_sets;       // Maximum user rule sets
    bool ptx_rules_loaded;             // True if PTX rules are loaded
};

// PTX rule function signature
typedef void (*DlPTXRuleFunc)(uint8_t* output, const uint8_t* input, uint32_t input_len, 
                               const DlRuleParams* params, uint32_t variant_idx);

// Rule function dispatch table
struct DlRuleFunctionTable {
    DlPTXRuleFunc ptx_functions[RULE_BUILTIN_COUNT];     // PTX rule functions
    bool ptx_available[RULE_BUILTIN_COUNT];              // PTX availability flags
};

// Host API for rule management
DlRuleManager* dl_create_rule_manager(void);
void dl_destroy_rule_manager(DlRuleManager* manager);

// Built-in rule management
bool dl_load_builtin_rules(DlRuleManager* manager);
bool dl_load_ptx_rules(DlRuleManager* manager);
const DlRuleSet* dl_get_builtin_best64(DlRuleManager* manager);

// User rule management
bool dl_load_user_rules_from_file(DlRuleManager* manager, const char* filepath, const char* name);
bool dl_load_user_rules_from_string(DlRuleManager* manager, const char* rules_string, const char* name);
bool dl_validate_rule_string(const char* rule);
const DlRuleSet* dl_get_user_rule_set(DlRuleManager* manager, const char* name);
bool dl_remove_user_rule_set(DlRuleManager* manager, const char* name);

// Rule compilation and optimization
bool dl_compile_hashcat_rule(const char* rule_string, DlCompiledRule* output);
bool dl_optimize_rule_set(DlRuleSet* rule_set);
uint64_t dl_calculate_rule_combinations(const DlRuleSet* rule_set);

// Rule selection and prioritization
DlRuleSet* dl_create_optimized_rule_set(DlRuleManager* manager, uint32_t max_rules, 
                                        const char* optimization_strategy);
bool dl_rank_rules_by_effectiveness(DlRuleSet* rule_set, const uint32_t* success_rates);

// GPU rule execution
void dl_execute_rule_batch_gpu(const DlRuleSet* rule_set, 
                               const uint8_t* input_words, const uint32_t* input_lengths,
                               uint8_t* output_candidates, uint32_t* output_lengths,
                               uint32_t word_count, uint32_t max_output_length);

// Rule analysis and profiling
struct DlRuleAnalysis {
    uint32_t rule_id;
    uint64_t applications;             // How many times applied
    uint64_t hits;                     // How many successful hits
    float hit_rate;                    // Success rate
    float avg_execution_time_us;       // Average execution time
    uint32_t avg_length_delta;         // Average length change
    uint32_t complexity_score;         // Computational complexity (1-10)
};

bool dl_analyze_rule_performance(DlRuleManager* manager, const char* wordlist_file, 
                                 DlRuleAnalysis** analysis, uint32_t* analysis_count);
void dl_print_rule_analysis(const DlRuleAnalysis* analysis, uint32_t count);

// Rule export/import
bool dl_export_rule_set(const DlRuleSet* rule_set, const char* filepath, const char* format);
bool dl_import_rule_set(DlRuleManager* manager, const char* filepath, const char* name);

// Memory management
void dl_free_rule_set(DlRuleSet* rule_set);
void dl_free_rule_analysis(DlRuleAnalysis* analysis);

// Constants for rule limits
#define DL_MAX_RULE_LENGTH 256         // Maximum characters in a rule string
#define DL_MAX_RULES_PER_SET 10000     // Maximum rules per rule set
#define DL_MAX_USER_RULE_SETS 100      // Maximum user rule sets
#define DL_MAX_RULE_PARAMS 16          // Maximum parameters per rule
#define DL_MAX_WORD_LENGTH 256         // Maximum word length for rules

// Best64 rule strings (for reference and compilation)
extern const char* DL_BEST64_RULES[64];

// Error codes
enum DlRuleError {
    DL_RULE_SUCCESS = 0,
    DL_RULE_ERROR_INVALID_SYNTAX = 1,
    DL_RULE_ERROR_TOO_LONG = 2,
    DL_RULE_ERROR_UNSUPPORTED = 3,
    DL_RULE_ERROR_MEMORY = 4,
    DL_RULE_ERROR_FILE_IO = 5,
    DL_RULE_ERROR_PTX_NOT_AVAILABLE = 6
};

// Utility functions
const char* dl_rule_error_string(enum DlRuleError error);
bool dl_is_rule_destructive(const DlCompiledRule* rule);  // Does rule potentially reduce keyspace?
uint32_t dl_estimate_rule_output_length(const DlCompiledRule* rule, uint32_t input_length);
float dl_estimate_rule_cost(const DlCompiledRule* rule);

// Configuration
struct DlRuleConfig {
    bool enable_ptx_rules;             // Use PTX rules when available
    bool enable_rule_chaining;         // Allow multiple rules per candidate
    uint32_t max_chain_length;         // Maximum rules in a chain
    bool enable_rule_caching;          // Cache compiled rules
    uint32_t max_cache_size_mb;        // Maximum cache size
    bool enable_rule_analysis;         // Enable performance analysis
    bool prefer_speed_over_coverage;   // Optimization preference
};

void dl_set_rule_config(DlRuleManager* manager, const DlRuleConfig* config);
DlRuleConfig dl_get_rule_config(DlRuleManager* manager);