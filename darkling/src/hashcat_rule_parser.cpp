#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <cstring>
#include "darkling_rule_manager.h"

// Hashcat Rule Parser
// Converts hashcat rule files to Darkling internal format

class HashcatRuleParser {
private:
    std::unordered_map<std::string, DlBuiltinRule> rule_mapping;
    
public:
    HashcatRuleParser() {
        initialize_rule_mapping();
    }
    
    void initialize_rule_mapping() {
        // Map hashcat rule strings to built-in rule IDs
        rule_mapping[":"] = RULE_BEST64_NOOP;
        rule_mapping["l"] = RULE_BEST64_LOWERCASE;
        rule_mapping["u"] = RULE_BEST64_UPPERCASE;
        rule_mapping["c"] = RULE_BEST64_CAPITALIZE;
        rule_mapping["C"] = RULE_BEST64_INVERT_CASE;
        rule_mapping["t"] = RULE_BEST64_TOGGLE_CASE;
        rule_mapping["r"] = RULE_BEST64_REVERSE;
        rule_mapping["d"] = RULE_BEST64_DUPLICATE;
        rule_mapping["f"] = RULE_BEST64_REFLECT;
        rule_mapping["{"] = RULE_BEST64_ROTATE_LEFT;
        rule_mapping["}"] = RULE_BEST64_ROTATE_RIGHT;
        rule_mapping["["] = RULE_BEST64_DELETE_FIRST;
        rule_mapping["]"] = RULE_BEST64_DELETE_LAST;
    }
    
    struct ParsedRule {
        DlRuleType type;
        DlBuiltinRule builtin_id;
        std::string rule_string;
        DlRuleParams params;
        bool is_valid;
    };
    
    ParsedRule parse_single_rule(const std::string& rule_str) {
        ParsedRule result;
        result.is_valid = false;
        result.type = RULE_TYPE_USER_INTERPRETED;
        result.rule_string = rule_str;
        memset(&result.params, 0, sizeof(result.params));
        
        if (rule_str.empty()) {
            return result;
        }
        
        // Check for direct built-in rule mapping
        auto it = rule_mapping.find(rule_str);
        if (it != rule_mapping.end()) {
            result.type = RULE_TYPE_BUILTIN_PTX;
            result.builtin_id = it->second;
            result.is_valid = true;
            result.params.variant_count = 1;
            result.params.max_length_delta = 0;
            
            // Set specific parameters for certain rules
            if (rule_str == "d") {
                result.params.max_length_delta = 256; // Duplicate can double length
            } else if (rule_str == "[" || rule_str == "]") {
                result.params.max_length_delta = -1; // Delete removes one character
            }
            
            return result;
        }
        
        // Parse parameterized rules
        if (rule_str.length() >= 2) {
            char first_char = rule_str[0];
            
            // Append character ($X)
            if (first_char == '$' && rule_str.length() == 2) {
                result.type = RULE_TYPE_BUILTIN_PTX;
                result.builtin_id = RULE_BEST64_APPEND_CHAR;
                result.is_valid = true;
                result.params.variant_count = 1;
                result.params.max_length_delta = 1;
                result.params.param_count = 1;
                result.params.params[0] = rule_str[1];
                return result;
            }
            
            // Prepend character (^X)
            if (first_char == '^' && rule_str.length() == 2) {
                result.type = RULE_TYPE_BUILTIN_PTX;
                result.builtin_id = RULE_BEST64_PREPEND_CHAR;
                result.is_valid = true;
                result.params.variant_count = 1;
                result.params.max_length_delta = 1;
                result.params.param_count = 1;
                result.params.params[0] = rule_str[1];
                return result;
            }
            
            // Character substitution (sXY)
            if (first_char == 's' && rule_str.length() == 3) {
                result.type = RULE_TYPE_BUILTIN_PTX;
                result.builtin_id = RULE_BEST64_REPLACE_CHAR;
                result.is_valid = true;
                result.params.variant_count = 1;
                result.params.max_length_delta = 0;
                result.params.param_count = 2;
                result.params.params[0] = rule_str[1]; // from char
                result.params.params[1] = rule_str[2]; // to char
                return result;
            }
            
            // Delete at position (DN)
            if (first_char == 'D' && rule_str.length() == 2 && std::isdigit(rule_str[1])) {
                result.type = RULE_TYPE_BUILTIN_PTX;
                result.builtin_id = RULE_BEST64_DELETE_AT_N;
                result.is_valid = true;
                result.params.variant_count = 1;
                result.params.max_length_delta = -1;
                result.params.param_count = 1;
                result.params.params[0] = rule_str[1] - '0';
                return result;
            }
            
            // Insert at position (iNX)
            if (first_char == 'i' && rule_str.length() == 3 && std::isdigit(rule_str[1])) {
                result.type = RULE_TYPE_BUILTIN_PTX;
                result.builtin_id = RULE_BEST64_INSERT_AT_N;
                result.is_valid = true;
                result.params.variant_count = 1;
                result.params.max_length_delta = 1;
                result.params.param_count = 2;
                result.params.params[0] = rule_str[1] - '0'; // position
                result.params.params[1] = rule_str[2];       // character
                return result;
            }
            
            // Overwrite at position (oNX)
            if (first_char == 'o' && rule_str.length() == 3 && std::isdigit(rule_str[1])) {
                result.type = RULE_TYPE_BUILTIN_PTX;
                result.builtin_id = RULE_BEST64_OVERWRITE_AT_N;
                result.is_valid = true;
                result.params.variant_count = 1;
                result.params.max_length_delta = 0;
                result.params.param_count = 2;
                result.params.params[0] = rule_str[1] - '0'; // position
                result.params.params[1] = rule_str[2];       // character
                return result;
            }
            
            // Truncate at position ('N)
            if (first_char == '\'' && rule_str.length() == 2 && std::isdigit(rule_str[1])) {
                result.type = RULE_TYPE_BUILTIN_PTX;
                result.builtin_id = RULE_BEST64_TRUNCATE_AT_N;
                result.is_valid = true;
                result.params.variant_count = 1;
                result.params.max_length_delta = -static_cast<int>(rule_str[1] - '0');
                result.params.param_count = 1;
                result.params.params[0] = rule_str[1] - '0';
                return result;
            }
            
            // Purge character (@X)
            if (first_char == '@' && rule_str.length() == 2) {
                result.type = RULE_TYPE_BUILTIN_PTX;
                result.builtin_id = RULE_BEST64_PURGE_CHAR;
                result.is_valid = true;
                result.params.variant_count = 1;
                result.params.max_length_delta = -1; // May remove characters
                result.params.param_count = 1;
                result.params.params[0] = rule_str[1];
                return result;
            }
        }
        
        // Handle complex rules that don't have direct built-in mappings
        if (rule_str.length() >= 3) {
            // Extract range (xNM)
            if (rule_str[0] == 'x' && rule_str.length() == 3 && 
                std::isdigit(rule_str[1]) && std::isdigit(rule_str[2])) {
                result.type = RULE_TYPE_BUILTIN_PTX;
                result.builtin_id = RULE_BEST64_EXTRACT_RANGE;
                result.is_valid = true;
                result.params.variant_count = 1;
                result.params.max_length_delta = -(rule_str[1] - '0'); // May reduce length
                result.params.param_count = 2;
                result.params.params[0] = rule_str[1] - '0'; // start position
                result.params.params[1] = rule_str[2] - '0'; // length
                return result;
            }
        }
        
        // If no built-in rule matches, mark as user-interpreted
        result.type = RULE_TYPE_USER_INTERPRETED;
        result.is_valid = true;
        result.params.variant_count = 1;
        result.params.max_length_delta = 0; // Conservative estimate
        
        return result;
    }
    
    std::vector<ParsedRule> parse_file(const std::string& filepath) {
        std::vector<ParsedRule> rules;
        std::ifstream file(filepath);
        
        if (!file.is_open()) {
            std::cerr << "Cannot open rule file: " << filepath << std::endl;
            return rules;
        }
        
        std::string line;
        int line_number = 0;
        
        while (std::getline(file, line)) {
            line_number++;
            
            // Remove leading/trailing whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') {
                continue;
            }
            
            ParsedRule parsed = parse_single_rule(line);
            if (parsed.is_valid) {
                rules.push_back(parsed);
            } else {
                std::cerr << "Warning: Invalid rule at line " << line_number 
                         << ": " << line << std::endl;
            }
        }
        
        file.close();
        return rules;
    }
    
    bool convert_to_darkling_rules(const std::vector<ParsedRule>& parsed_rules,
                                   DlRuleSet* rule_set, const std::string& name) {
        if (parsed_rules.empty()) {
            return false;
        }
        
        rule_set->rule_count = parsed_rules.size();
        rule_set->rules = new DlCompiledRule[rule_set->rule_count];
        
        strncpy(rule_set->name, name.c_str(), sizeof(rule_set->name) - 1);
        rule_set->name[sizeof(rule_set->name) - 1] = '\0';
        
        strncpy(rule_set->description, "Converted from hashcat rule file", 
                sizeof(rule_set->description) - 1);
        rule_set->description[sizeof(rule_set->description) - 1] = '\0';
        
        rule_set->is_builtin = false;
        rule_set->total_combinations = 0;
        
        for (size_t i = 0; i < parsed_rules.size(); ++i) {
            const ParsedRule& parsed = parsed_rules[i];
            DlCompiledRule& compiled = rule_set->rules[i];
            
            compiled.type = parsed.type;
            compiled.rule_id = (parsed.type == RULE_TYPE_BUILTIN_PTX) ? parsed.builtin_id : i;
            compiled.params = parsed.params;
            
            strncpy(compiled.rule_string, parsed.rule_string.c_str(), 
                    sizeof(compiled.rule_string) - 1);
            compiled.rule_string[sizeof(compiled.rule_string) - 1] = '\0';
            
            // Estimate computational cost
            compiled.estimated_cost = estimate_rule_cost(parsed.rule_string);
            compiled.success_rate_permille = 50; // Conservative estimate
            
            rule_set->total_combinations += parsed.params.variant_count;
        }
        
        return true;
    }
    
private:
    float estimate_rule_cost(const std::string& rule_str) {
        if (rule_str.empty()) return 0.0f;
        
        char first_char = rule_str[0];
        
        // Simple rules (low cost)
        if (rule_str == ":" || rule_str == "l" || rule_str == "u") return 0.1f;
        if (first_char == '$' || first_char == '^') return 0.2f;
        if (first_char == 's') return 0.3f;
        
        // Medium complexity
        if (rule_str == "c" || rule_str == "C" || rule_str == "t") return 0.4f;
        if (rule_str == "[" || rule_str == "]") return 0.3f;
        if (first_char == 'D' || first_char == '@') return 0.4f;
        
        // High complexity
        if (rule_str == "r" || rule_str == "{" || rule_str == "}") return 0.5f;
        if (rule_str == "d" || rule_str == "f") return 0.8f;
        if (first_char == 'i' || first_char == 'o' || first_char == 'x') return 0.6f;
        
        return 0.5f; // Default
    }
};

// C interface for the parser
extern "C" {
    
bool dl_parse_hashcat_rules(const char* filepath, DlRuleSet* rule_set, const char* name) {
    if (!filepath || !rule_set || !name) {
        return false;
    }
    
    HashcatRuleParser parser;
    auto parsed_rules = parser.parse_file(std::string(filepath));
    
    if (parsed_rules.empty()) {
        return false;
    }
    
    return parser.convert_to_darkling_rules(parsed_rules, rule_set, std::string(name));
}

bool dl_validate_hashcat_rule_string(const char* rule_str) {
    if (!rule_str) {
        return false;
    }
    
    HashcatRuleParser parser;
    auto parsed = parser.parse_single_rule(std::string(rule_str));
    return parsed.is_valid;
}

uint32_t dl_count_hashcat_rules(const char* filepath) {
    if (!filepath) {
        return 0;
    }
    
    HashcatRuleParser parser;
    auto parsed_rules = parser.parse_file(std::string(filepath));
    return static_cast<uint32_t>(parsed_rules.size());
}

} // extern "C"