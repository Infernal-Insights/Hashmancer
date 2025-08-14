#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdio>
#include <cassert>
#include <unistd.h>
#include <sys/wait.h>

// Integration test for CLI compatibility with Hashmancer worker

void create_test_files() {
    // Create test hash file
    std::ofstream hash_file("test_hashes.txt");
    hash_file << "5d41402abc4b2a76b9719d911017c592\n";  // MD5 of "hello"
    hash_file << "aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d\n";  // SHA1 of "hello" 
    hash_file << "8846f7eaee8fb117ad06bdd830b7586c\n";  // NTLM of "hello"
    hash_file.close();
    
    // Create test rule file
    std::ofstream rule_file("test_rules.txt");
    rule_file << "# Test rules\n";
    rule_file << ":\n";      // no-op
    rule_file << "l\n";      // lowercase
    rule_file << "u\n";      // uppercase
    rule_file << "$1\n";     // append 1
    rule_file << "^@\n";     // prepend @
    rule_file.close();
    
    // Create test wordlist
    std::ofstream wordlist("test_wordlist.txt");
    wordlist << "hello\n";
    wordlist << "world\n";
    wordlist << "test\n";
    wordlist.close();
}

void cleanup_test_files() {
    std::remove("test_hashes.txt");
    std::remove("test_rules.txt");
    std::remove("test_wordlist.txt");
    std::remove("test_output.txt");
}

bool test_hashcat_compatibility() {
    std::cout << "Testing hashcat CLI compatibility..." << std::endl;
    
    // Test command that mimics what Hashmancer worker sends
    std::vector<std::string> test_commands = {
        // Dictionary attack with rules (what worker sends)
        "./main -m 0 test_hashes.txt -a 0 test_wordlist.txt -r test_rules.txt --quiet --outfile test_output.txt",
        
        // Mask attack
        "./main -m 0 test_hashes.txt -a 3 ?d?d?d?d --quiet",
        
        // Status reporting
        "./main -m 0 test_hashes.txt -a 0 test_wordlist.txt --status --status-timer 1 --quiet",
        
        // JSON status
        "./main -m 0 test_hashes.txt -a 0 test_wordlist.txt --status-json --quiet"
    };
    
    for (const auto& cmd : test_commands) {
        std::cout << "Testing: " << cmd << std::endl;
        
        int result = system(cmd.c_str());
        if (WEXITSTATUS(result) != 0) {
            std::cerr << "Command failed: " << cmd << std::endl;
            return false;
        }
        
        std::cout << "✓ Command succeeded" << std::endl;
    }
    
    return true;
}

bool test_worker_expected_interface() {
    std::cout << "Testing worker-expected interface..." << std::endl;
    
    // These are the exact commands the worker sends
    std::vector<std::string> worker_commands = {
        // Dictionary + rules attack (most common)
        "./main -m 0 test_hashes.txt --shard test_wordlist.txt --rules test_rules.txt -d 1 --quiet --status --status-json --status-timer 10 --outfile test_output.txt --outfile-format 2",
        
        // Mask attack
        "./main -m 0 test_hashes.txt -a 3 ?d?d?d?d?d?d -d 1 --quiet --status --status-json",
        
        // Dictionary only
        "./main -m 0 test_hashes.txt -a 0 test_wordlist.txt -d 1 --quiet"
    };
    
    for (const auto& cmd : worker_commands) {
        std::cout << "Testing worker command: " << cmd << std::endl;
        
        int result = system(cmd.c_str());
        if (WEXITSTATUS(result) != 0) {
            std::cerr << "Worker command failed: " << cmd << std::endl;
            return false;
        }
        
        std::cout << "✓ Worker command succeeded" << std::endl;
    }
    
    return true;
}

bool test_rule_parsing() {
    std::cout << "Testing rule file parsing..." << std::endl;
    
    // Create comprehensive rule file
    std::ofstream rule_file("comprehensive_rules.txt");
    rule_file << "# Comprehensive test rules\n";
    rule_file << ":\n";        // no-op
    rule_file << "l\n";        // lowercase
    rule_file << "u\n";        // uppercase
    rule_file << "c\n";        // capitalize
    rule_file << "r\n";        // reverse
    rule_file << "d\n";        // duplicate
    rule_file << "$0\n";       // append 0
    rule_file << "$1\n";       // append 1
    rule_file << "^!\n";       // prepend !
    rule_file << "^@\n";       // prepend @
    rule_file << "se3\n";      // substitute e->3
    rule_file << "sa@\n";      // substitute a->@
    rule_file << "D0\n";       // delete at position 0
    rule_file << "D1\n";       // delete at position 1
    rule_file.close();
    
    std::string cmd = "./main -m 0 test_hashes.txt -a 0 test_wordlist.txt -r comprehensive_rules.txt --quiet";
    int result = system(cmd.c_str());
    
    std::remove("comprehensive_rules.txt");
    
    if (WEXITSTATUS(result) != 0) {
        std::cerr << "Rule parsing test failed" << std::endl;
        return false;
    }
    
    std::cout << "✓ Rule parsing test passed" << std::endl;
    return true;
}

bool test_output_formats() {
    std::cout << "Testing output formats..." << std::endl;
    
    std::vector<std::pair<int, std::string>> formats = {
        {1, "hash:plain"},
        {2, "plain"},
        {3, "hex_plain"}
    };
    
    for (const auto& fmt : formats) {
        std::string outfile = "test_output_" + std::to_string(fmt.first) + ".txt";
        std::string cmd = "./main -m 0 test_hashes.txt -a 0 test_wordlist.txt --outfile " + 
                         outfile + " --outfile-format " + std::to_string(fmt.first) + " --quiet";
        
        int result = system(cmd.c_str());
        if (WEXITSTATUS(result) != 0) {
            std::cerr << "Output format test failed for format " << fmt.first << std::endl;
            return false;
        }
        
        std::remove(outfile.c_str());
        std::cout << "✓ Output format " << fmt.second << " test passed" << std::endl;
    }
    
    return true;
}

int main() {
    std::cout << "=== Darkling CLI Integration Test Suite ===" << std::endl;
    
    // Check if main executable exists
    if (access("./main", X_OK) != 0) {
        std::cerr << "Error: ./main executable not found. Please build first." << std::endl;
        return 1;
    }
    
    create_test_files();
    
    bool all_passed = true;
    
    try {
        all_passed &= test_hashcat_compatibility();
        all_passed &= test_worker_expected_interface();
        all_passed &= test_rule_parsing();
        all_passed &= test_output_formats();
        
        if (all_passed) {
            std::cout << "\n✅ All CLI integration tests passed!" << std::endl;
            std::cout << "Darkling is compatible with Hashmancer worker interface." << std::endl;
        } else {
            std::cout << "\n❌ Some tests failed!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        all_passed = false;
    }
    
    cleanup_test_files();
    
    return all_passed ? 0 : 1;
}