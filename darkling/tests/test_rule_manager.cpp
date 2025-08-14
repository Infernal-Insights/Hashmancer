#include <iostream>
#include <cassert>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include "darkling_rule_manager.h"

// Test suite for the rule manager functionality

void test_rule_manager_creation() {
    std::cout << "Testing rule manager creation..." << std::endl;
    
    DlRuleManager* manager = dl_create_rule_manager();
    assert(manager != nullptr);
    
    dl_destroy_rule_manager(manager);
    std::cout << "✓ Rule manager creation test passed" << std::endl;
}

void test_builtin_rules_loading() {
    std::cout << "Testing built-in rules loading..." << std::endl;
    
    DlRuleManager* manager = dl_create_rule_manager();
    assert(manager != nullptr);
    
    bool result = dl_load_builtin_rules(manager);
    assert(result == true);
    
    bool ptx_result = dl_load_ptx_rules(manager);
    assert(ptx_result == true);
    
    dl_destroy_rule_manager(manager);
    std::cout << "✓ Built-in rules loading test passed" << std::endl;
}

void test_rule_validation() {
    std::cout << "Testing rule validation..." << std::endl;
    
    // Valid rules
    assert(dl_validate_rule_string(":") == true);          // no-op
    assert(dl_validate_rule_string("l") == true);          // lowercase
    assert(dl_validate_rule_string("u") == true);          // uppercase
    assert(dl_validate_rule_string("c") == true);          // capitalize
    assert(dl_validate_rule_string("r") == true);          // reverse
    assert(dl_validate_rule_string("d") == true);          // duplicate
    assert(dl_validate_rule_string("$0") == true);         // append digit
    assert(dl_validate_rule_string("^a") == true);         // prepend char
    assert(dl_validate_rule_string("se3") == true);        // substitution
    assert(dl_validate_rule_string("D0") == true);         // delete at pos
    
    // Invalid rules
    assert(dl_validate_rule_string("") == false);          // empty
    assert(dl_validate_rule_string("x") == false);         // unsupported
    assert(dl_validate_rule_string("$") == false);         // incomplete
    assert(dl_validate_rule_string("s1") == false);        // incomplete substitution
    assert(dl_validate_rule_string("Da") == false);        // invalid delete pos
    
    std::cout << "✓ Rule validation test passed" << std::endl;
}

void test_user_rule_loading() {
    std::cout << "Testing user rule loading..." << std::endl;
    
    DlRuleManager* manager = dl_create_rule_manager();
    assert(manager != nullptr);
    
    // Create a temporary rule file
    std::string temp_file = "/tmp/test_rules.txt";
    std::ofstream file(temp_file);
    file << "# Test rule set\n";
    file << ":\n";      // no-op
    file << "l\n";      // lowercase
    file << "u\n";      // uppercase
    file << "$1\n";     // append 1
    file << "^@\n";     // prepend @
    file << "se3\n";    // leet e->3
    file.close();
    
    bool result = dl_load_user_rules_from_file(manager, temp_file.c_str(), "test_rules");
    assert(result == true);
    
    // Clean up
    std::remove(temp_file.c_str());
    dl_destroy_rule_manager(manager);
    
    std::cout << "✓ User rule loading test passed" << std::endl;
}

void test_rule_batch_execution() {
    std::cout << "Testing rule batch execution..." << std::endl;
    
    DlRuleManager* manager = dl_create_rule_manager();
    assert(manager != nullptr);
    
    bool load_result = dl_load_builtin_rules(manager);
    assert(load_result == true);
    
    // Create test input data
    const uint32_t word_count = 3;
    const uint32_t max_word_len = 16;
    const uint32_t rule_count = 2;
    
    // Test words: "test", "pass", "word"
    uint8_t input_words[word_count * max_word_len];
    uint32_t input_lengths[word_count] = {4, 4, 4};
    memset(input_words, 0, sizeof(input_words));
    
    // Copy test words
    memcpy(input_words + 0 * max_word_len, "test", 4);
    memcpy(input_words + 1 * max_word_len, "pass", 4);
    memcpy(input_words + 2 * max_word_len, "word", 4);
    
    // Create mock rule set with simple rules
    DlRuleSet rule_set;
    rule_set.rule_count = 2;
    rule_set.rules = new DlCompiledRule[2];
    
    // Rule 1: no-op (:)
    rule_set.rules[0].type = RULE_TYPE_USER_INTERPRETED;
    rule_set.rules[0].rule_id = 0;
    strcpy(rule_set.rules[0].rule_string, ":");
    rule_set.rules[0].params.max_length_delta = 0;
    
    // Rule 2: lowercase (l)
    rule_set.rules[1].type = RULE_TYPE_USER_INTERPRETED;
    rule_set.rules[1].rule_id = 1;
    strcpy(rule_set.rules[1].rule_string, "l");
    rule_set.rules[1].params.max_length_delta = 0;
    
    // Allocate output buffers
    uint8_t* output_candidates = new uint8_t[word_count * rule_count * max_word_len];
    uint32_t* output_lengths = new uint32_t[word_count * rule_count];
    
    // Execute rules (this will test the GPU kernel)
    dl_execute_rule_batch_gpu(&rule_set, input_words, input_lengths,
                              output_candidates, output_lengths,
                              word_count, max_word_len);
    
    // Verify some basic outputs
    assert(output_lengths[0] == 4);  // "test" with no-op should be length 4
    assert(output_lengths[1] == 4);  // "test" with lowercase should be length 4
    
    // Clean up
    delete[] rule_set.rules;
    delete[] output_candidates;
    delete[] output_lengths;
    dl_destroy_rule_manager(manager);
    
    std::cout << "✓ Rule batch execution test passed" << std::endl;
}

void test_error_handling() {
    std::cout << "Testing error handling..." << std::endl;
    
    // Test error string function
    const char* error_str = dl_rule_error_string(DL_RULE_ERROR_INVALID_SYNTAX);
    assert(error_str != nullptr);
    assert(strlen(error_str) > 0);
    
    // Test with invalid file path
    DlRuleManager* manager = dl_create_rule_manager();
    bool result = dl_load_user_rules_from_file(manager, "/nonexistent/path/rules.txt", "test");
    assert(result == false);
    
    dl_destroy_rule_manager(manager);
    std::cout << "✓ Error handling test passed" << std::endl;
}

void test_memory_management() {
    std::cout << "Testing memory management..." << std::endl;
    
    // Create and destroy multiple managers
    for (int i = 0; i < 10; ++i) {
        DlRuleManager* manager = dl_create_rule_manager();
        assert(manager != nullptr);
        
        dl_load_builtin_rules(manager);
        dl_destroy_rule_manager(manager);
    }
    
    std::cout << "✓ Memory management test passed" << std::endl;
}

void run_performance_test() {
    std::cout << "Running performance test..." << std::endl;
    
    DlRuleManager* manager = dl_create_rule_manager();
    dl_load_builtin_rules(manager);
    
    const uint32_t word_count = 1000;
    const uint32_t max_word_len = 32;
    
    // Generate test data
    uint8_t* input_words = new uint8_t[word_count * max_word_len];
    uint32_t* input_lengths = new uint32_t[word_count];
    
    for (uint32_t i = 0; i < word_count; ++i) {
        snprintf((char*)(input_words + i * max_word_len), max_word_len, "testword%u", i);
        input_lengths[i] = strlen((char*)(input_words + i * max_word_len));
    }
    
    // Create simple rule set
    DlRuleSet rule_set;
    rule_set.rule_count = 1;
    rule_set.rules = new DlCompiledRule[1];
    rule_set.rules[0].type = RULE_TYPE_USER_INTERPRETED;
    strcpy(rule_set.rules[0].rule_string, "l");
    
    uint8_t* output_candidates = new uint8_t[word_count * max_word_len];
    uint32_t* output_lengths = new uint32_t[word_count];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    dl_execute_rule_batch_gpu(&rule_set, input_words, input_lengths,
                              output_candidates, output_lengths,
                              word_count, max_word_len);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Processed " << word_count << " words in " << duration.count() << " μs" << std::endl;
    std::cout << "Rate: " << (word_count * 1000000.0 / duration.count()) << " words/second" << std::endl;
    
    // Clean up
    delete[] input_words;
    delete[] input_lengths;
    delete[] output_candidates;
    delete[] output_lengths;
    delete[] rule_set.rules;
    dl_destroy_rule_manager(manager);
    
    std::cout << "✓ Performance test completed" << std::endl;
}

int main() {
    std::cout << "=== Darkling Rule Manager Test Suite ===" << std::endl;
    
    try {
        test_rule_manager_creation();
        test_builtin_rules_loading();
        test_rule_validation();
        test_user_rule_loading();
        test_rule_batch_execution();
        test_error_handling();
        test_memory_management();
        
        // Performance test (optional)
        run_performance_test();
        
        std::cout << "\n✅ All rule manager tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}