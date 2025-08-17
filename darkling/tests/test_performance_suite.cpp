#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <fstream>
#include <random>
#include <algorithm>

#include "../include/gpu_manager.h"
#include "../include/rule_analytics.h"
#include "../include/checkpoint_manager.h"

// Comprehensive Performance Testing and Validation Suite

class PerformanceTestSuite {
public:
    struct TestResult {
        std::string test_name;
        bool passed;
        double execution_time_seconds;
        double performance_metric;
        std::string performance_unit;
        std::string notes;
    };
    
    struct BenchmarkResult {
        std::string component;
        double baseline_performance;
        double current_performance;
        double improvement_factor;
        bool meets_requirements;
    };

private:
    std::vector<TestResult> test_results_;
    std::vector<BenchmarkResult> benchmark_results_;
    
public:
    // Multi-GPU Performance Tests
    TestResult test_multi_gpu_detection() {
        auto start = std::chrono::high_resolution_clock::now();
        
        GPUManager gpu_manager;
        bool success = gpu_manager.initialize();
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        
        TestResult result;
        result.test_name = "Multi-GPU Detection";
        result.passed = success && gpu_manager.get_gpu_count() > 0;
        result.execution_time_seconds = elapsed;
        result.performance_metric = gpu_manager.get_gpu_count();
        result.performance_unit = "GPUs detected";
        
        if (gpu_manager.is_multi_gpu_available()) {
            result.notes = "Multi-GPU support available";
        } else {
            result.notes = "Single GPU system";
        }
        
        return result;
    }
    
    TestResult test_workload_distribution() {
        GPUManager gpu_manager;
        if (!gpu_manager.initialize()) {
            return {"Workload Distribution", false, 0.0, 0.0, "N/A", "GPU manager failed to initialize"};
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<int> devices;
        auto available_gpus = gpu_manager.get_available_gpus();
        for (const auto& gpu : available_gpus) {
            devices.push_back(gpu.device_id);
        }
        
        uint64_t total_keyspace = 1000000000ULL; // 1 billion candidates
        auto distributions = gpu_manager.distribute_workload(total_keyspace, devices);
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        
        TestResult result;
        result.test_name = "Workload Distribution";
        result.passed = distributions.size() == devices.size();
        result.execution_time_seconds = elapsed;
        result.performance_metric = distributions.size();
        result.performance_unit = "workloads distributed";
        
        // Validate distribution
        uint64_t total_distributed = 0;
        for (const auto& dist : distributions) {
            total_distributed += (dist.keyspace_end - dist.keyspace_start);
        }
        
        if (total_distributed != total_keyspace) {
            result.passed = false;
            result.notes = "Keyspace distribution mismatch";
        } else {
            result.notes = "Keyspace distributed correctly";
        }
        
        return result;
    }
    
    TestResult test_gpu_memory_management() {
        GPUManager gpu_manager;
        if (!gpu_manager.initialize()) {
            return {"GPU Memory Management", false, 0.0, 0.0, "N/A", "GPU manager failed to initialize"};
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        bool success = true;
        size_t total_allocated = 0;
        std::vector<void*> allocations;
        
        // Test memory allocation on each device
        auto available_gpus = gpu_manager.get_available_gpus();
        for (const auto& gpu : available_gpus) {
            size_t alloc_size = 100 * 1024 * 1024; // 100MB
            void* ptr = nullptr;
            
            if (gpu_manager.allocate_gpu_memory(gpu.device_id, alloc_size, &ptr)) {
                allocations.push_back(ptr);
                total_allocated += alloc_size;
            } else {
                success = false;
                break;
            }
        }
        
        // Free allocated memory
        for (size_t i = 0; i < allocations.size() && i < available_gpus.size(); ++i) {
            gpu_manager.free_gpu_memory(available_gpus[i].device_id, allocations[i]);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        
        TestResult result;
        result.test_name = "GPU Memory Management";
        result.passed = success;
        result.execution_time_seconds = elapsed;
        result.performance_metric = total_allocated / (1024.0 * 1024.0);
        result.performance_unit = "MB allocated/freed";
        result.notes = success ? "Memory operations successful" : "Memory allocation failed";
        
        return result;
    }
    
    // Rule Analytics Performance Tests
    TestResult test_rule_analytics_performance() {
        auto start = std::chrono::high_resolution_clock::now();
        
        RuleAnalytics analytics;
        bool success = analytics.initialize();
        
        if (!success) {
            return {"Rule Analytics Performance", false, 0.0, 0.0, "N/A", "Analytics initialization failed"};
        }
        
        // Simulate rule applications
        std::vector<std::string> test_rules = {
            ":", "l", "u", "c", "r", "d", "$1", "$2", "^@", "se3"
        };
        
        int operations = 0;
        for (int i = 0; i < 1000; ++i) {
            for (const auto& rule : test_rules) {
                analytics.record_rule_application(
                    rule, "password", "password123", (i % 10 == 0), "md5", "test", 0.1
                );
                operations++;
            }
        }
        
        // Test performance analysis
        auto performance_data = analytics.get_rule_performance_ranking("success_rate", 10);
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        
        TestResult result;
        result.test_name = "Rule Analytics Performance";
        result.passed = !performance_data.empty();
        result.execution_time_seconds = elapsed;
        result.performance_metric = operations / elapsed;
        result.performance_unit = "operations/second";
        result.notes = "Processed " + std::to_string(operations) + " rule applications";
        
        return result;
    }
    
    TestResult test_smart_rule_selection() {
        RuleAnalytics analytics;
        if (!analytics.initialize()) {
            return {"Smart Rule Selection", false, 0.0, 0.0, "N/A", "Analytics initialization failed"};
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        SmartRuleSelector selector(&analytics);
        
        SmartRuleSelector::SelectionCriteria criteria;
        criteria.hash_type = "md5";
        criteria.wordlist_path = "test_wordlist.txt";
        criteria.time_budget_seconds = 300.0;
        criteria.target_success_rate = 0.05;
        criteria.max_rules = 20;
        criteria.optimization_goal = "efficiency";
        
        auto selected_rules = selector.select_optimal_rules(criteria);
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        
        TestResult result;
        result.test_name = "Smart Rule Selection";
        result.passed = !selected_rules.empty() && selected_rules.size() <= criteria.max_rules;
        result.execution_time_seconds = elapsed;
        result.performance_metric = selected_rules.size();
        result.performance_unit = "rules selected";
        result.notes = "Selected " + std::to_string(selected_rules.size()) + " optimal rules";
        
        return result;
    }
    
    // Checkpoint System Tests
    TestResult test_checkpoint_creation_performance() {
        CheckpointManager checkpoint_manager;
        if (!checkpoint_manager.initialize()) {
            return {"Checkpoint Creation", false, 0.0, 0.0, "N/A", "Checkpoint manager initialization failed"};
        }
        
        // Create test checkpoint data
        CheckpointData test_data;
        test_data.job_id = "test_job_001";
        test_data.attack_type = "dictionary";
        test_data.hash_type = "md5";
        test_data.keyspace_total = 1000000;
        test_data.keyspace_processed = 500000;
        test_data.candidates_tested = 750000;
        test_data.hashes_cracked = 42;
        test_data.progress_percentage = 50.0;
        test_data.wordlist_path = "/test/wordlist.txt";
        test_data.target_hashes = {"hash1", "hash2", "hash3"};
        test_data.cracked_passwords = {"password1", "password2"};
        test_data.current_word_index = 250000;
        test_data.active_devices = {0, 1};
        test_data.device_positions = {125000, 125000};
        
        auto start = std::chrono::high_resolution_clock::now();
        
        bool success = checkpoint_manager.create_checkpoint("test_job_001", test_data);
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        
        TestResult result;
        result.test_name = "Checkpoint Creation";
        result.passed = success;
        result.execution_time_seconds = elapsed;
        result.performance_metric = 1.0 / elapsed;
        result.performance_unit = "checkpoints/second";
        result.notes = success ? "Checkpoint created successfully" : "Checkpoint creation failed";
        
        // Cleanup
        if (success) {
            checkpoint_manager.delete_checkpoint("test_job_001");
        }
        
        return result;
    }
    
    TestResult test_checkpoint_restoration_performance() {
        CheckpointManager checkpoint_manager;
        if (!checkpoint_manager.initialize()) {
            return {"Checkpoint Restoration", false, 0.0, 0.0, "N/A", "Checkpoint manager initialization failed"};
        }
        
        // Create checkpoint first
        CheckpointData test_data;
        test_data.job_id = "test_job_002";
        test_data.attack_type = "dictionary";
        test_data.hash_type = "md5";
        test_data.keyspace_total = 1000000;
        test_data.keyspace_processed = 500000;
        
        if (!checkpoint_manager.create_checkpoint("test_job_002", test_data)) {
            return {"Checkpoint Restoration", false, 0.0, 0.0, "N/A", "Failed to create test checkpoint"};
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        CheckpointData restored_data = checkpoint_manager.load_checkpoint("test_job_002");
        bool success = (restored_data.job_id == "test_job_002");
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        
        TestResult result;
        result.test_name = "Checkpoint Restoration";
        result.passed = success;
        result.execution_time_seconds = elapsed;
        result.performance_metric = 1.0 / elapsed;
        result.performance_unit = "restorations/second";
        result.notes = success ? "Checkpoint restored successfully" : "Checkpoint restoration failed";
        
        // Cleanup
        checkpoint_manager.delete_checkpoint("test_job_002");
        
        return result;
    }
    
    // Integration Performance Tests
    TestResult test_end_to_end_performance() {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Initialize all systems
        GPUManager gpu_manager;
        RuleAnalytics analytics;
        CheckpointManager checkpoint_manager;
        
        bool gpu_init = gpu_manager.initialize();
        bool analytics_init = analytics.initialize();
        bool checkpoint_init = checkpoint_manager.initialize();
        
        if (!gpu_init || !analytics_init || !checkpoint_init) {
            return {"End-to-End Performance", false, 0.0, 0.0, "N/A", "System initialization failed"};
        }
        
        // Simulate a complete job workflow
        SmartRuleSelector selector(&analytics);
        SmartRuleSelector::SelectionCriteria criteria;
        criteria.hash_type = "md5";
        criteria.time_budget_seconds = 60.0;
        criteria.max_rules = 10;
        
        auto selected_rules = selector.select_optimal_rules(criteria);
        
        // Create checkpoint
        CheckpointData checkpoint_data;
        checkpoint_data.job_id = "integration_test_job";
        checkpoint_data.attack_type = "dictionary";
        checkpoint_data.hash_type = "md5";
        checkpoint_data.keyspace_total = 100000;
        
        bool checkpoint_created = checkpoint_manager.create_checkpoint("integration_test_job", checkpoint_data);
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        
        TestResult result;
        result.test_name = "End-to-End Performance";
        result.passed = !selected_rules.empty() && checkpoint_created;
        result.execution_time_seconds = elapsed;
        result.performance_metric = selected_rules.size();
        result.performance_unit = "rules processed";
        result.notes = "Complete workflow executed successfully";
        
        // Cleanup
        if (checkpoint_created) {
            checkpoint_manager.delete_checkpoint("integration_test_job");
        }
        
        return result;
    }
    
    // Benchmark against baseline performance
    BenchmarkResult benchmark_against_baseline() {
        // These would be established baseline metrics
        const double baseline_gpu_detection_time = 0.1; // 100ms
        const double baseline_workload_distribution_time = 0.01; // 10ms
        const double baseline_checkpoint_creation_time = 0.05; // 50ms
        
        // Run current performance tests
        auto gpu_test = test_multi_gpu_detection();
        auto workload_test = test_workload_distribution();
        auto checkpoint_test = test_checkpoint_creation_performance();
        
        BenchmarkResult result;
        result.component = "Overall Performance";
        result.baseline_performance = baseline_gpu_detection_time + baseline_workload_distribution_time + baseline_checkpoint_creation_time;
        result.current_performance = gpu_test.execution_time_seconds + workload_test.execution_time_seconds + checkpoint_test.execution_time_seconds;
        result.improvement_factor = result.baseline_performance / result.current_performance;
        result.meets_requirements = result.improvement_factor >= 0.8; // Within 20% of baseline
        
        return result;
    }
    
public:
    void run_all_tests() {
        std::cout << "=== Hashmancer Performance Test Suite ===" << std::endl;
        std::cout << std::endl;
        
        // Run all performance tests
        test_results_.push_back(test_multi_gpu_detection());
        test_results_.push_back(test_workload_distribution());
        test_results_.push_back(test_gpu_memory_management());
        test_results_.push_back(test_rule_analytics_performance());
        test_results_.push_back(test_smart_rule_selection());
        test_results_.push_back(test_checkpoint_creation_performance());
        test_results_.push_back(test_checkpoint_restoration_performance());
        test_results_.push_back(test_end_to_end_performance());
        
        // Run benchmark
        benchmark_results_.push_back(benchmark_against_baseline());
        
        // Print results
        print_results();
    }
    
    void print_results() {
        std::cout << "Test Results:" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        int passed = 0, total = 0;
        for (const auto& result : test_results_) {
            total++;
            if (result.passed) passed++;
            
            std::string status = result.passed ? "✓ PASS" : "✗ FAIL";
            std::printf("%-30s %s %8.3fs %10.2f %s\n",
                       result.test_name.c_str(),
                       status.c_str(),
                       result.execution_time_seconds,
                       result.performance_metric,
                       result.performance_unit.c_str());
            
            if (!result.notes.empty()) {
                std::printf("    %s\n", result.notes.c_str());
            }
        }
        
        std::cout << std::string(80, '-') << std::endl;
        std::printf("Tests Passed: %d/%d (%.1f%%)\n", passed, total, (passed * 100.0) / total);
        
        if (!benchmark_results_.empty()) {
            std::cout << std::endl << "Benchmark Results:" << std::endl;
            std::cout << std::string(80, '-') << std::endl;
            
            for (const auto& bench : benchmark_results_) {
                std::string status = bench.meets_requirements ? "✓ MEETS" : "✗ BELOW";
                std::printf("%-30s %s %.2fx improvement\n",
                           bench.component.c_str(),
                           status.c_str(),
                           bench.improvement_factor);
            }
        }
        
        std::cout << std::endl;
        
        if (passed == total && !benchmark_results_.empty() && benchmark_results_[0].meets_requirements) {
            std::cout << "🎉 All tests passed and performance meets requirements!" << std::endl;
            std::cout << "Hashmancer immediate impact improvements are working correctly." << std::endl;
        } else {
            std::cout << "⚠️  Some tests failed or performance is below expectations." << std::endl;
            std::cout << "Review the results above for specific issues." << std::endl;
        }
    }
};

int main() {
    PerformanceTestSuite test_suite;
    test_suite.run_all_tests();
    return 0;
}