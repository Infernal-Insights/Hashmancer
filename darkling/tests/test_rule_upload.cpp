#include <iostream>
#include <cassert>
#include <cstring>
#include <fstream>
#include <string>
#include "darkling_rule_manager.h"

// Test suite for rule upload and validation functionality

// Declare C functions for rule upload handling
extern "C" {
    struct DlRuleUploadResult {
        bool success;
        char error_message[256];
        uint32_t rule_count;
        uint32_t complexity_score;
        uint32_t warning_count;
        char warnings[10][128];
    };
    
    DlRuleUploadResult* dl_validate_rule_upload(const char* rules_content, const char* format);
    bool dl_save_user_rules(const char* rules_content, const char* rule_set_name, const char* output_path);
    uint32_t dl_estimate_rule_performance_impact(const char* rules_content);
}

void test_valid_rule_upload() {
    std::cout << "Testing valid rule upload..." << std::endl;
    
    const char* valid_rules = R"(
# Test rule set
:
l
u
c
r
d
$0
$1
$2
^!
^@
se3
sa@
si1
D0
D1
)";
    
    DlRuleUploadResult* result = dl_validate_rule_upload(valid_rules, "hashcat");
    assert(result != nullptr);
    assert(result->success == true);
    assert(result->rule_count > 0);
    assert(strlen(result->error_message) == 0);
    
    std::cout << "✓ Validated " << result->rule_count << " rules successfully" << std::endl;
    std::cout << "✓ Complexity score: " << result->complexity_score << std::endl;
    if (result->warning_count > 0) {
        std::cout << "✓ Warnings: " << result->warning_count << std::endl;
    }
    
    std::cout << "✓ Valid rule upload test passed" << std::endl;
}

void test_invalid_rule_upload() {
    std::cout << "Testing invalid rule upload..." << std::endl;
    
    // Test with dangerous patterns
    const char* dangerous_rules = R"(
l
u
exec("rm -rf /")
$0
)";
    
    DlRuleUploadResult* result = dl_validate_rule_upload(dangerous_rules, "hashcat");
    assert(result != nullptr);
    assert(result->success == false);
    assert(strlen(result->error_message) > 0);
    
    std::cout << "✓ Correctly rejected dangerous rule: " << result->error_message << std::endl;
    
    // Test with invalid syntax
    const char* invalid_syntax = R"(
l
u
invalid_rule_xyz
$0
)";
    
    result = dl_validate_rule_upload(invalid_syntax, "hashcat");
    assert(result != nullptr);
    assert(result->success == false);
    assert(strlen(result->error_message) > 0);
    
    std::cout << "✓ Correctly rejected invalid syntax: " << result->error_message << std::endl;
    
    // Test with incomplete rules
    const char* incomplete_rules = R"(
l
u
$
se
^
)";
    
    result = dl_validate_rule_upload(incomplete_rules, "hashcat");
    assert(result != nullptr);
    assert(result->success == false);
    
    std::cout << "✓ Correctly rejected incomplete rules" << std::endl;
    std::cout << "✓ Invalid rule upload test passed" << std::endl;
}

void test_rule_complexity_estimation() {
    std::cout << "Testing rule complexity estimation..." << std::endl;
    
    // Simple rules (low complexity)
    const char* simple_rules = R"(
:
l
u
c
)";
    
    uint32_t impact = dl_estimate_rule_performance_impact(simple_rules);
    assert(impact < 50);  // Should be low impact
    std::cout << "✓ Simple rules impact: " << impact << "%" << std::endl;
    
    // Complex rules (high complexity)
    const char* complex_rules = R"(
d
f
r
x12
*34
)";
    
    impact = dl_estimate_rule_performance_impact(complex_rules);
    assert(impact > 30);  // Should be higher impact
    std::cout << "✓ Complex rules impact: " << impact << "%" << std::endl;
    
    std::cout << "✓ Rule complexity estimation test passed" << std::endl;
}

void test_rule_saving() {
    std::cout << "Testing rule saving..." << std::endl;
    
    const char* test_rules = R"(
# Test rule set for saving
:
l
u
$1
^@
se3
)";
    
    std::string temp_file = "/tmp/test_saved_rules.txt";
    
    bool result = dl_save_user_rules(test_rules, "test_upload", temp_file.c_str());
    assert(result == true);
    
    // Verify file was created and contains rules
    std::ifstream file(temp_file);
    assert(file.is_open());
    
    std::string content;
    std::string line;
    while (std::getline(file, line)) {
        content += line + "\n";
    }
    file.close();
    
    assert(content.find("test_upload") != std::string::npos);
    assert(content.find("l") != std::string::npos);
    assert(content.find("u") != std::string::npos);
    
    // Clean up
    std::remove(temp_file.c_str());
    
    std::cout << "✓ Rule saving test passed" << std::endl;
}

void test_duplicate_detection() {
    std::cout << "Testing duplicate rule detection..." << std::endl;
    
    const char* duplicate_rules = R"(
l
u
l
c
u
$1
$1
)";
    
    DlRuleUploadResult* result = dl_validate_rule_upload(duplicate_rules, "hashcat");
    assert(result != nullptr);
    assert(result->success == true);  // Should still be valid
    assert(result->warning_count > 0);  // But should have warnings
    
    // Check if warnings mention duplicates
    bool found_duplicate_warning = false;
    for (uint32_t i = 0; i < result->warning_count; ++i) {
        if (strstr(result->warnings[i], "duplicate") != nullptr || 
            strstr(result->warnings[i], "Duplicate") != nullptr) {
            found_duplicate_warning = true;
            break;
        }
    }
    assert(found_duplicate_warning);
    
    std::cout << "✓ Duplicate detection test passed" << std::endl;
}

void test_format_support() {
    std::cout << "Testing different rule formats..." << std::endl;
    
    const char* hashcat_rules = R"(
l
u
$1
^@
se3
)";
    
    // Test hashcat format
    DlRuleUploadResult* result = dl_validate_rule_upload(hashcat_rules, "hashcat");
    assert(result != nullptr);
    assert(result->success == true);
    std::cout << "✓ Hashcat format validation passed" << std::endl;
    
    // Test auto-detection
    result = dl_validate_rule_upload(hashcat_rules, "auto");
    assert(result != nullptr);
    assert(result->success == true);
    std::cout << "✓ Auto format detection passed" << std::endl;
    
    std::cout << "✓ Format support test passed" << std::endl;
}

void test_large_rule_upload() {
    std::cout << "Testing large rule upload..." << std::endl;
    
    // Generate a large but valid rule set
    std::string large_rules = "# Large rule set\n";
    for (int i = 0; i < 500; ++i) {
        large_rules += ":\n";  // no-op rules
    }
    
    DlRuleUploadResult* result = dl_validate_rule_upload(large_rules.c_str(), "hashcat");
    assert(result != nullptr);
    assert(result->success == true);
    assert(result->rule_count == 500);
    
    std::cout << "✓ Large rule upload test passed (500 rules)" << std::endl;
    
    // Test exceeding the limit
    std::string oversized_rules = "# Oversized rule set\n";
    for (int i = 0; i < 1200; ++i) {  // Exceed typical limits
        oversized_rules += ":\n";
    }
    
    result = dl_validate_rule_upload(oversized_rules.c_str(), "hashcat");
    assert(result != nullptr);
    // Should either succeed with warnings or fail gracefully
    
    std::cout << "✓ Large rule upload validation completed" << std::endl;
}

void test_edge_cases() {
    std::cout << "Testing edge cases..." << std::endl;
    
    // Empty content
    DlRuleUploadResult* result = dl_validate_rule_upload("", "hashcat");
    assert(result != nullptr);
    assert(result->success == false);
    std::cout << "✓ Empty content correctly rejected" << std::endl;
    
    // Only comments
    result = dl_validate_rule_upload("# Only comments\n# No actual rules\n", "hashcat");
    assert(result != nullptr);
    assert(result->success == false);
    std::cout << "✓ Comment-only content correctly rejected" << std::endl;
    
    // Null pointer
    result = dl_validate_rule_upload(nullptr, "hashcat");
    assert(result != nullptr);
    assert(result->success == false);
    std::cout << "✓ Null content correctly handled" << std::endl;
    
    // Very long rule
    std::string long_rule(300, 'a');  // 300 character rule
    result = dl_validate_rule_upload(long_rule.c_str(), "hashcat");
    assert(result != nullptr);
    assert(result->success == false);
    std::cout << "✓ Oversized rule correctly rejected" << std::endl;
    
    std::cout << "✓ Edge cases test passed" << std::endl;
}

void test_performance_impact_estimation() {
    std::cout << "Testing performance impact estimation..." << std::endl;
    
    // Very fast rules
    const char* fast_rules = ":\nl\nu\nc\n";
    uint32_t impact = dl_estimate_rule_performance_impact(fast_rules);
    assert(impact < 30);
    std::cout << "✓ Fast rules impact: " << impact << "%" << std::endl;
    
    // Medium speed rules
    const char* medium_rules = "$1\n^@\nse3\nsa@\nr\n";
    impact = dl_estimate_rule_performance_impact(medium_rules);
    assert(impact >= 20 && impact <= 60);
    std::cout << "✓ Medium rules impact: " << impact << "%" << std::endl;
    
    // Slow rules
    const char* slow_rules = "d\nf\nx12\n*34\n";
    impact = dl_estimate_rule_performance_impact(slow_rules);
    assert(impact > 40);
    std::cout << "✓ Slow rules impact: " << impact << "%" << std::endl;
    
    std::cout << "✓ Performance impact estimation test passed" << std::endl;
}

int main() {
    std::cout << "=== Darkling Rule Upload Test Suite ===" << std::endl;
    
    try {
        test_valid_rule_upload();
        test_invalid_rule_upload();
        test_rule_complexity_estimation();
        test_rule_saving();
        test_duplicate_detection();
        test_format_support();
        test_large_rule_upload();
        test_edge_cases();
        test_performance_impact_estimation();
        
        std::cout << "\n✅ All rule upload tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}