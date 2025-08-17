#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <atomic>
#include <memory>
#include <mutex>
#include "darkling_rule_manager.h"

// Advanced Rule Analytics and Intelligence System
// Tracks rule effectiveness and provides ML-driven optimization

struct RulePerformanceData {
    std::string rule_string;           // Original rule (e.g., "l", "$1", "se3")
    DlBuiltinRule rule_id;            // Internal rule ID
    uint64_t total_applications;       // How many times applied
    uint64_t successful_hits;          // How many resulted in cracks
    uint64_t failed_attempts;         // How many produced no results
    double success_rate;              // hits / applications
    double average_execution_time_ms; // Average time per application
    uint64_t total_candidates_generated; // Total password candidates created
    uint64_t unique_patterns_hit;     // Number of unique password patterns cracked
    
    // Context-specific metrics
    std::unordered_map<std::string, uint64_t> success_by_hash_type; // MD5, SHA1, etc.
    std::unordered_map<std::string, uint64_t> success_by_wordlist; // wordlist source
    std::unordered_map<int, uint64_t> success_by_length; // password length distribution
    
    // Time-based analysis
    std::chrono::system_clock::time_point first_used;
    std::chrono::system_clock::time_point last_used;
    std::chrono::system_clock::time_point last_success;
    
    // Performance metrics
    double cpu_efficiency;            // Success rate per CPU cycle
    double memory_efficiency;         // Success rate per MB used
    double power_efficiency;          // Success rate per watt consumed
};

struct RulePattern {
    std::string pattern_regex;        // Regex describing cracked passwords
    std::vector<std::string> contributing_rules; // Rules that crack this pattern
    uint64_t occurrences;            // How often this pattern appears
    double pattern_strength;         // Estimated password strength
    std::vector<std::string> example_passwords; // Sample passwords matching pattern
};

struct RuleRecommendation {
    std::string rule_string;
    double confidence_score;         // 0.0 - 1.0 confidence in recommendation
    double estimated_success_rate;   // Predicted success rate for this target
    std::string reasoning;           // Human-readable explanation
    int priority;                   // 1-10 priority ranking
    double estimated_time_to_first_crack; // Seconds
    double estimated_completion_time; // Seconds for full rule execution
};

class RuleAnalytics {
public:
    RuleAnalytics();
    ~RuleAnalytics();
    
    // Initialize with database connection for persistence
    bool initialize(const std::string& database_path = "");
    
    // Track rule usage and results
    void record_rule_application(
        const std::string& rule_string,
        const std::string& input_word,
        const std::string& output_candidate,
        bool was_successful_crack,
        const std::string& hash_type = "unknown",
        const std::string& wordlist_source = "unknown",
        double execution_time_ms = 0.0
    );
    
    void record_batch_results(
        const std::vector<std::string>& rules_used,
        uint64_t total_candidates_generated,
        uint64_t successful_cracks,
        const std::vector<std::string>& cracked_passwords,
        const std::string& context_info = ""
    );
    
    // Analysis and insights
    std::vector<RulePerformanceData> get_rule_performance_ranking(
        const std::string& sort_by = "success_rate", // success_rate, efficiency, recent_performance
        int limit = 50
    );
    
    RulePerformanceData get_rule_performance(const std::string& rule_string);
    
    std::vector<RulePattern> identify_password_patterns(
        const std::vector<std::string>& cracked_passwords
    );
    
    // Intelligent recommendations
    std::vector<RuleRecommendation> recommend_rules_for_target(
        const std::vector<std::string>& sample_hashes,
        const std::string& hash_type,
        const std::string& wordlist_path,
        int max_recommendations = 20
    );
    
    std::vector<RuleRecommendation> recommend_rules_for_pattern(
        const std::string& suspected_pattern, // e.g., "corporate_passwords", "personal_passwords"
        const std::string& domain_context = "" // e.g., "company.com", "gaming", "social_media"
    );
    
    // Adaptive rule selection
    std::vector<std::string> create_adaptive_ruleset(
        const std::vector<std::string>& base_rules,
        const std::string& target_context,
        double success_rate_threshold = 0.01, // Minimum 1% success rate
        int max_rules = 100
    );
    
    // Performance optimization
    std::vector<std::string> optimize_rule_order(
        const std::vector<std::string>& rules,
        const std::string& optimization_strategy = "time_to_first_crack" // or "total_coverage"
    );
    
    bool should_skip_rule(
        const std::string& rule_string,
        const std::string& context,
        double time_budget_seconds
    );
    
    // Machine learning features
    struct MLFeatures {
        std::vector<double> rule_complexity_features;    // Numeric features describing rule complexity
        std::vector<double> target_context_features;     // Features describing the target context
        std::vector<double> historical_performance_features; // Past performance in similar contexts
        std::vector<double> wordlist_compatibility_features; // How well rule works with wordlist
    };
    
    MLFeatures extract_features_for_prediction(
        const std::string& rule_string,
        const std::string& target_context,
        const std::string& wordlist_info
    );
    
    double predict_rule_success_rate(const MLFeatures& features);
    
    // Learning and adaptation
    void update_models_from_recent_data();
    bool export_training_data(const std::string& output_path);
    bool import_training_data(const std::string& input_path);
    
    // Analytics and reporting
    struct AnalyticsReport {
        uint64_t total_rules_tracked;
        uint64_t total_applications;
        uint64_t total_successful_cracks;
        double overall_success_rate;
        std::vector<RulePerformanceData> top_performing_rules;
        std::vector<RulePattern> discovered_patterns;
        std::vector<std::string> underperforming_rules;
        std::string report_period;
        std::chrono::system_clock::time_point generated_at;
    };
    
    AnalyticsReport generate_performance_report(
        const std::string& time_period = "last_30_days" // last_day, last_week, last_month, all_time
    );
    
    // Real-time monitoring
    void start_real_time_monitoring();
    void stop_real_time_monitoring();
    
    struct RealTimeMetrics {
        double current_success_rate;
        double trending_success_rate;        // Direction of success rate change
        std::string most_effective_rule_now;
        std::string least_effective_rule_now;
        uint64_t cracks_per_minute;
        std::vector<std::string> emerging_patterns;
    };
    
    RealTimeMetrics get_real_time_metrics();

private:
    // Data storage
    std::unordered_map<std::string, RulePerformanceData> rule_performance_cache_;
    std::vector<RulePattern> discovered_patterns_;
    std::mutex analytics_mutex_;
    
    // Database connection for persistence
    std::unique_ptr<class DatabaseConnection> db_connection_;
    
    // Machine learning models (simplified interface)
    std::unique_ptr<class MLModel> success_prediction_model_;
    std::unique_ptr<class MLModel> pattern_recognition_model_;
    
    // Real-time monitoring
    std::atomic<bool> monitoring_active_;
    std::unique_ptr<std::thread> monitoring_thread_;
    
    // Internal analysis methods
    double calculate_rule_complexity_score(const std::string& rule_string);
    std::vector<std::string> extract_rule_components(const std::string& rule_string);
    bool is_rule_compatible_with_wordlist(const std::string& rule_string, const std::string& wordlist_path);
    double estimate_execution_time(const std::string& rule_string, uint64_t wordlist_size);
    
    // Pattern analysis
    std::string classify_password_pattern(const std::string& password);
    double calculate_pattern_strength(const std::string& pattern);
    std::vector<std::string> suggest_rules_for_pattern(const std::string& pattern);
    
    // Database operations
    bool save_performance_data();
    bool load_performance_data();
    bool create_database_schema();
    
    // ML model operations
    bool train_success_prediction_model();
    bool train_pattern_recognition_model();
    std::vector<double> normalize_features(const std::vector<double>& features);
};

// Smart Rule Selector - High-level interface for rule selection
class SmartRuleSelector {
public:
    SmartRuleSelector(RuleAnalytics* analytics);
    
    struct SelectionCriteria {
        std::string hash_type;
        std::string wordlist_path;
        double time_budget_seconds;
        double target_success_rate;
        int max_rules;
        std::string optimization_goal; // "speed", "coverage", "efficiency"
        std::vector<std::string> required_rules; // Rules that must be included
        std::vector<std::string> excluded_rules; // Rules to avoid
    };
    
    std::vector<std::string> select_optimal_rules(const SelectionCriteria& criteria);
    
    // Dynamic rule adjustment during execution
    bool should_continue_with_rule(
        const std::string& rule_string,
        uint64_t candidates_processed,
        uint64_t cracks_found,
        double time_elapsed_seconds
    );
    
    std::string suggest_next_rule(
        const std::vector<std::string>& completed_rules,
        const std::vector<std::string>& cracked_passwords,
        double remaining_time_budget
    );

private:
    RuleAnalytics* analytics_;
    
    double calculate_rule_value_score(
        const std::string& rule,
        const SelectionCriteria& criteria
    );
    
    std::vector<std::string> apply_optimization_strategy(
        std::vector<std::string> candidate_rules,
        const SelectionCriteria& criteria
    );
};

// Utility functions for rule analysis
namespace rule_analytics_utils {
    // Parse rule components for analysis
    struct RuleComponents {
        std::string operation;          // "append", "prepend", "substitute", etc.
        std::vector<std::string> parameters; // Characters or positions
        int complexity_score;          // 1-10 complexity rating
        bool is_destructive;          // Can reduce password length
        bool is_generative;           // Can increase password length
    };
    
    RuleComponents parse_rule(const std::string& rule_string);
    
    // Password pattern analysis
    std::string identify_password_pattern(const std::string& password);
    std::vector<std::string> extract_password_features(const std::string& password);
    double calculate_password_entropy(const std::string& password);
    
    // Rule compatibility analysis
    bool are_rules_compatible(const std::string& rule1, const std::string& rule2);
    std::vector<std::string> find_conflicting_rules(const std::vector<std::string>& rules);
    std::vector<std::string> suggest_complementary_rules(const std::string& base_rule);
    
    // Performance estimation
    double estimate_rule_performance(
        const std::string& rule,
        const std::string& wordlist_sample,
        const std::string& hash_type
    );
    
    uint64_t estimate_candidate_count(
        const std::vector<std::string>& rules,
        uint64_t wordlist_size
    );
}