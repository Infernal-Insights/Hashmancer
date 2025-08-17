#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_set>
#include <regex>
#include <algorithm>
#include <cstring>
#include "darkling_rule_manager.h"

// Advanced rule validation and upload system for user-provided rules
// Ensures security, performance, and compatibility with Darkling engine

class RuleUploadHandler {
private:
    std::unordered_set<std::string> dangerous_patterns;
    std::unordered_set<std::string> supported_rule_types;
    uint32_t max_rules_per_upload;
    uint32_t max_rule_complexity;
    
public:
    RuleUploadHandler() {
        initialize_security_patterns();
        initialize_supported_rules();
        max_rules_per_upload = 1000;
        max_rule_complexity = 10;
    }
    
    void initialize_security_patterns() {
        // Patterns that could be dangerous or resource-intensive
        dangerous_patterns.insert(".*\\$\\{.*\\}.*");      // Variable expansion
        dangerous_patterns.insert(".*\\|.*");             // Shell pipes
        dangerous_patterns.insert(".*&&.*");              // Command chaining
        dangerous_patterns.insert(".*\\;.*");             // Command termination
        dangerous_patterns.insert(".*\\`.*");             // Command substitution
        dangerous_patterns.insert(".*exec.*");            // Execution commands
        dangerous_patterns.insert(".*eval.*");            // Code evaluation
        dangerous_patterns.insert(".*system.*");          // System calls
        dangerous_patterns.insert(".*fork.*");            // Process forking
        dangerous_patterns.insert(".*/dev/.*");           // Device access
        dangerous_patterns.insert(".*/proc/.*");          // Process access
        dangerous_patterns.insert(".*/sys/.*");           // System access
        dangerous_patterns.insert(".*\\.\\./.*");         // Directory traversal
    }
    
    void initialize_supported_rules() {
        // Core hashcat rule syntax support
        supported_rule_types.insert(":");        // no-op
        supported_rule_types.insert("l");        // lowercase
        supported_rule_types.insert("u");        // uppercase
        supported_rule_types.insert("c");        // capitalize
        supported_rule_types.insert("C");        // invert case
        supported_rule_types.insert("t");        // toggle case
        supported_rule_types.insert("r");        // reverse
        supported_rule_types.insert("d");        // duplicate
        supported_rule_types.insert("f");        // reflect
        supported_rule_types.insert("{");        // rotate left
        supported_rule_types.insert("}");        // rotate right
        supported_rule_types.insert("[");        // delete first
        supported_rule_types.insert("]");        // delete last
        supported_rule_types.insert("k");        // swap first two
        supported_rule_types.insert("K");        // swap last two
        supported_rule_types.insert("*");        // swap at N
        supported_rule_types.insert("L");        // bitwise shift left
        supported_rule_types.insert("R");        // bitwise shift right
        supported_rule_types.insert("+");        // increment at N
        supported_rule_types.insert("-");        // decrement at N
        supported_rule_types.insert(".");        // replace at N
        supported_rule_types.insert(",");        // insert at N
        supported_rule_types.insert("y");        // duplicate block
        supported_rule_types.insert("Y");        // duplicate block N times
        supported_rule_types.insert("E");        // title case
        supported_rule_types.insert("q");        // duplicate every char
        supported_rule_types.insert("z");        // uppercase first, lowercase rest
        supported_rule_types.insert("Z");        // lowercase first, uppercase rest
        
        // Parameterized rules ($ ^ s @ D etc) handled separately
    }
    
    struct ValidationResult {
        bool is_valid;
        std::string error_message;
        uint32_t complexity_score;
        std::vector<std::string> warnings;
        std::vector<std::string> preprocessed_rules;
    };
    
    ValidationResult validate_rule_upload(const std::string& rules_content, 
                                         const std::string& format = "hashcat") {
        ValidationResult result;
        result.is_valid = true;
        result.complexity_score = 0;
        
        // Step 1: Parse rules from content
        std::vector<std::string> raw_rules = parse_rules_from_content(rules_content, format);
        
        if (raw_rules.empty()) {
            result.is_valid = false;
            result.error_message = "No valid rules found in upload";
            return result;
        }
        
        if (raw_rules.size() > max_rules_per_upload) {
            result.is_valid = false;
            result.error_message = "Too many rules in upload (max: " + std::to_string(max_rules_per_upload) + ")";
            return result;
        }
        
        // Step 2: Validate each rule
        for (const auto& rule : raw_rules) {
            auto rule_validation = validate_single_rule(rule);
            
            if (!rule_validation.is_valid) {
                result.is_valid = false;
                result.error_message = "Invalid rule '" + rule + "': " + rule_validation.error_message;
                return result;
            }
            
            result.complexity_score += rule_validation.complexity_score;
            result.warnings.insert(result.warnings.end(), 
                                 rule_validation.warnings.begin(), 
                                 rule_validation.warnings.end());
            
            result.preprocessed_rules.push_back(rule_validation.preprocessed_rules[0]);
        }
        
        // Step 3: Check overall complexity
        if (result.complexity_score > max_rule_complexity * raw_rules.size()) {
            result.warnings.push_back("High complexity rule set may impact performance");
        }
        
        // Step 4: Check for redundant rules
        auto duplicate_check = check_for_duplicates(result.preprocessed_rules);
        if (!duplicate_check.empty()) {
            result.warnings.push_back("Duplicate rules detected: " + duplicate_check);
        }
        
        return result;
    }
    
    ValidationResult validate_single_rule(const std::string& rule) {
        ValidationResult result;
        result.is_valid = true;
        result.complexity_score = 1;
        
        // Security validation
        for (const auto& pattern : dangerous_patterns) {
            if (std::regex_match(rule, std::regex(pattern))) {
                result.is_valid = false;
                result.error_message = "Rule contains potentially dangerous pattern";
                return result;
            }
        }
        
        // Length check
        if (rule.length() == 0 || rule.length() > DL_MAX_RULE_LENGTH) {
            result.is_valid = false;
            result.error_message = "Rule length invalid (0 or > " + std::to_string(DL_MAX_RULE_LENGTH) + ")";
            return result;
        }
        
        // Syntax validation
        auto syntax_result = validate_rule_syntax(rule);
        if (!syntax_result.first) {
            result.is_valid = false;
            result.error_message = syntax_result.second;
            return result;
        }
        
        // Complexity estimation
        result.complexity_score = estimate_rule_complexity(rule);
        
        // Performance warnings
        if (result.complexity_score > 7) {
            result.warnings.push_back("High complexity rule may impact performance: " + rule);
        }
        
        // Normalize and preprocess rule
        std::string normalized = normalize_rule(rule);
        result.preprocessed_rules.push_back(normalized);
        
        return result;
    }
    
    std::pair<bool, std::string> validate_rule_syntax(const std::string& rule) {
        if (rule.empty()) {
            return {false, "Empty rule"};
        }
        
        // Single character rules
        if (rule.length() == 1) {
            if (supported_rule_types.count(rule)) {
                return {true, ""};
            } else {
                return {false, "Unsupported single character rule: " + rule};
            }
        }
        
        // Multi-character rules
        char first_char = rule[0];
        
        // Append/prepend rules ($X, ^X)
        if ((first_char == '$' || first_char == '^') && rule.length() == 2) {
            char param = rule[1];
            if (std::isprint(param) && param != ' ') {
                return {true, ""};
            } else {
                return {false, "Invalid character parameter: " + std::string(1, param)};
            }
        }
        
        // Substitution rules (sXY)
        if (first_char == 's' && rule.length() == 3) {
            char from_char = rule[1];
            char to_char = rule[2];
            if (std::isprint(from_char) && std::isprint(to_char)) {
                return {true, ""};
            } else {
                return {false, "Invalid substitution characters"};
            }
        }
        
        // Delete at position (DN)
        if (first_char == 'D' && rule.length() == 2) {
            char pos = rule[1];
            if (std::isdigit(pos)) {
                return {true, ""};
            } else {
                return {false, "Invalid position for delete rule: " + std::string(1, pos)};
            }
        }
        
        // Insert at position (iNX)
        if (first_char == 'i' && rule.length() == 3) {
            char pos = rule[1];
            char ch = rule[2];
            if (std::isdigit(pos) && std::isprint(ch)) {
                return {true, ""};
            } else {
                return {false, "Invalid insert rule parameters"};
            }
        }
        
        // Overwrite at position (oNX)
        if (first_char == 'o' && rule.length() == 3) {
            char pos = rule[1];
            char ch = rule[2];
            if (std::isdigit(pos) && std::isprint(ch)) {
                return {true, ""};
            } else {
                return {false, "Invalid overwrite rule parameters"};
            }
        }
        
        // Truncate at position ('N)
        if (first_char == '\'' && rule.length() == 2) {
            char pos = rule[1];
            if (std::isdigit(pos)) {
                return {true, ""};
            } else {
                return {false, "Invalid truncate position: " + std::string(1, pos)};
            }
        }
        
        // Extract substring (xNM)
        if (first_char == 'x' && rule.length() == 3) {
            char start = rule[1];
            char len = rule[2];
            if (std::isdigit(start) && std::isdigit(len)) {
                return {true, ""};
            } else {
                return {false, "Invalid extract rule parameters"};
            }
        }
        
        // Purge character (@X)
        if (first_char == '@' && rule.length() == 2) {
            char ch = rule[1];
            if (std::isprint(ch)) {
                return {true, ""};
            } else {
                return {false, "Invalid purge character: " + std::string(1, ch)};
            }
        }
        
        // Position-based rules (*NM, pN, etc.)
        if (first_char == '*' && rule.length() == 3) {
            char pos1 = rule[1];
            char pos2 = rule[2];
            if (std::isdigit(pos1) && std::isdigit(pos2)) {
                return {true, ""};
            }
        }
        
        if (first_char == 'p' && rule.length() == 2) {
            char pos = rule[1];
            if (std::isdigit(pos)) {
                return {true, ""};
            }
        }
        
        // Multi-character operations (complex rules)
        if (rule.length() > 3) {
            // More complex rule validation would go here
            // For now, conservatively reject very long rules
            if (rule.length() > 10) {
                return {false, "Rule too complex (length > 10)"};
            }
        }
        
        return {false, "Unrecognized rule syntax: " + rule};
    }
    
    uint32_t estimate_rule_complexity(const std::string& rule) {
        if (rule.empty()) return 0;
        
        char first_char = rule[0];
        
        // Simple operations
        if (rule == ":" || rule == "l" || rule == "u") return 1;
        if (rule == "c" || rule == "C" || rule == "t") return 2;
        if (first_char == '$' || first_char == '^') return 2;
        if (first_char == 's') return 3;
        
        // Medium complexity
        if (rule == "r" || rule == "{" || rule == "}") return 4;
        if (rule == "[" || rule == "]") return 3;
        if (first_char == 'D' || first_char == '@') return 3;
        
        // High complexity
        if (rule == "d" || rule == "f") return 6;
        if (first_char == 'i' || first_char == 'o') return 5;
        if (first_char == 'x' || first_char == '*') return 7;
        
        // Very high complexity
        if (rule.length() > 5) return 9;
        
        return 4; // Default complexity
    }
    
    std::string normalize_rule(const std::string& rule) {
        // Remove any whitespace
        std::string normalized = rule;
        normalized.erase(std::remove_if(normalized.begin(), normalized.end(), ::isspace), normalized.end());
        
        // Convert to lowercase for case-insensitive operations where appropriate
        // (but preserve case for character parameters)
        
        return normalized;
    }
    
    std::vector<std::string> parse_rules_from_content(const std::string& content, const std::string& format) {
        std::vector<std::string> rules;
        
        if (format == "hashcat" || format == "auto") {
            // Parse hashcat rule format (one rule per line)
            std::istringstream iss(content);
            std::string line;
            
            while (std::getline(iss, line)) {
                // Skip comments and empty lines
                if (line.empty() || line[0] == '#') continue;
                
                // Remove trailing whitespace
                line.erase(line.find_last_not_of(" \t\r\n") + 1);
                
                if (!line.empty()) {
                    rules.push_back(line);
                }
            }
        }
        
        return rules;
    }
    
    std::string check_for_duplicates(const std::vector<std::string>& rules) {
        std::unordered_set<std::string> seen;
        std::vector<std::string> duplicates;
        
        for (const auto& rule : rules) {
            if (seen.count(rule)) {
                duplicates.push_back(rule);
            } else {
                seen.insert(rule);
            }
        }
        
        if (duplicates.empty()) {
            return "";
        }
        
        std::string result = "Found " + std::to_string(duplicates.size()) + " duplicates: ";
        for (size_t i = 0; i < std::min(duplicates.size(), size_t(5)); ++i) {
            if (i > 0) result += ", ";
            result += duplicates[i];
        }
        if (duplicates.size() > 5) {
            result += ", ...";
        }
        
        return result;
    }
    
    bool save_validated_rules(const std::vector<std::string>& rules, 
                            const std::string& rule_set_name,
                            const std::string& output_path) {
        std::ofstream file(output_path);
        if (!file.is_open()) {
            return false;
        }
        
        // Write header
        file << "# Rule set: " << rule_set_name << "\n";
        file << "# Generated by Darkling rule upload handler\n";
        file << "# Total rules: " << rules.size() << "\n";
        file << "\n";
        
        // Write rules
        for (const auto& rule : rules) {
            file << rule << "\n";
        }
        
        file.close();
        return true;
    }
};

// C interface for integration with main rule manager
extern "C" {
    
struct DlRuleUploadResult {
    bool success;
    char error_message[256];
    uint32_t rule_count;
    uint32_t complexity_score;
    uint32_t warning_count;
    char warnings[10][128];  // Up to 10 warnings, 128 chars each
};

DlRuleUploadResult* dl_validate_rule_upload(const char* rules_content, const char* format) {
    static DlRuleUploadResult result;
    memset(&result, 0, sizeof(result));
    
    try {
        RuleUploadHandler handler;
        auto validation = handler.validate_rule_upload(
            std::string(rules_content), 
            format ? std::string(format) : "hashcat"
        );
        
        result.success = validation.is_valid;
        result.rule_count = validation.preprocessed_rules.size();
        result.complexity_score = validation.complexity_score;
        result.warning_count = std::min(validation.warnings.size(), size_t(10));
        
        if (!validation.is_valid) {
            strncpy(result.error_message, validation.error_message.c_str(), 255);
            result.error_message[255] = '\0';
        }
        
        // Copy warnings
        for (size_t i = 0; i < result.warning_count; ++i) {
            strncpy(result.warnings[i], validation.warnings[i].c_str(), 127);
            result.warnings[i][127] = '\0';
        }
        
    } catch (const std::exception& e) {
        result.success = false;
        strncpy(result.error_message, e.what(), 255);
        result.error_message[255] = '\0';
    }
    
    return &result;
}

bool dl_save_user_rules(const char* rules_content, const char* rule_set_name, const char* output_path) {
    try {
        RuleUploadHandler handler;
        auto validation = handler.validate_rule_upload(std::string(rules_content), "hashcat");
        
        if (!validation.is_valid) {
            return false;
        }
        
        return handler.save_validated_rules(
            validation.preprocessed_rules,
            std::string(rule_set_name),
            std::string(output_path)
        );
        
    } catch (const std::exception& e) {
        return false;
    }
}

uint32_t dl_estimate_rule_performance_impact(const char* rules_content) {
    try {
        RuleUploadHandler handler;
        auto validation = handler.validate_rule_upload(std::string(rules_content), "hashcat");
        
        if (!validation.is_valid) {
            return 100; // Maximum impact for invalid rules
        }
        
        // Return percentage impact estimate (0-100)
        uint32_t avg_complexity = validation.complexity_score / std::max(validation.preprocessed_rules.size(), size_t(1));
        return std::min(avg_complexity * 10, uint32_t(100));
        
    } catch (const std::exception& e) {
        return 100;
    }
}

} // extern "C"