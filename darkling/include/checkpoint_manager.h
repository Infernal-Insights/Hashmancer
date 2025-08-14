#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <atomic>
#include <mutex>
#include <fstream>

// Advanced Checkpoint and Resume System
// Enables graceful recovery from interruptions and efficient job management

struct CheckpointData {
    std::string job_id;
    std::string attack_type;          // "dictionary", "mask", "hybrid", "rule"
    std::string hash_type;            // "md5", "sha1", "ntlm"
    
    // Progress tracking
    uint64_t keyspace_total;
    uint64_t keyspace_processed;
    uint64_t candidates_tested;
    uint64_t hashes_cracked;
    double progress_percentage;
    
    // Attack parameters
    std::string wordlist_path;
    std::string rules_path;
    std::string mask_pattern;
    std::vector<std::string> target_hashes;
    std::vector<std::string> cracked_passwords;
    
    // State information
    uint64_t current_word_index;      // For dictionary attacks
    uint64_t current_rule_index;      // For rule-based attacks  
    uint64_t current_mask_position;   // For mask attacks
    std::vector<uint64_t> device_positions; // Per-GPU progress
    
    // Performance metrics
    double average_hash_rate;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point last_checkpoint;
    std::chrono::duration<double> total_runtime;
    std::chrono::duration<double> estimated_remaining;
    
    // System state
    std::vector<int> active_devices;
    std::string worker_node_id;
    std::string checkpoint_version;
    
    // Recovery information
    std::vector<uint8_t> gpu_memory_snapshot; // Critical GPU state
    std::string error_context;        // Last error before checkpoint
    bool requires_full_restart;      // Some states can't be resumed
};

class CheckpointManager {
public:
    CheckpointManager();
    ~CheckpointManager();
    
    // Initialization
    bool initialize(const std::string& checkpoint_directory = "/tmp/hashmancer_checkpoints");
    bool set_auto_checkpoint_interval(int seconds);
    bool enable_crash_recovery(bool enabled = true);
    
    // Checkpoint creation
    bool create_checkpoint(
        const std::string& job_id,
        const CheckpointData& data
    );
    
    bool auto_checkpoint(
        const std::string& job_id,
        const CheckpointData& data
    );
    
    // Checkpoint restoration
    bool has_checkpoint(const std::string& job_id);
    CheckpointData load_checkpoint(const std::string& job_id);
    
    bool resume_job(
        const std::string& job_id,
        CheckpointData& restored_data
    );
    
    // Checkpoint management
    std::vector<std::string> list_available_checkpoints();
    bool delete_checkpoint(const std::string& job_id);
    bool cleanup_old_checkpoints(int max_age_hours = 24);
    
    // Advanced features
    bool create_incremental_checkpoint(
        const std::string& job_id,
        const CheckpointData& data
    );
    
    bool verify_checkpoint_integrity(const std::string& job_id);
    
    bool compress_checkpoint(const std::string& job_id);
    bool decompress_checkpoint(const std::string& job_id);
    
    // Multi-node coordination
    bool sync_checkpoint_to_shared_storage(const std::string& job_id);
    bool load_checkpoint_from_shared_storage(const std::string& job_id);
    
    // Performance optimization
    bool enable_fast_checkpointing(bool enabled = true);
    bool set_checkpoint_compression_level(int level = 6); // 1-9
    
    // Monitoring and statistics
    struct CheckpointStats {
        uint64_t total_checkpoints_created;
        uint64_t total_checkpoints_restored;
        uint64_t total_jobs_recovered;
        double average_checkpoint_time_ms;
        double average_restore_time_ms;
        uint64_t total_checkpoint_storage_bytes;
        double checkpoint_success_rate;
    };
    
    CheckpointStats get_statistics();
    
    // Event callbacks
    typedef std::function<void(const std::string& job_id, bool success)> CheckpointCallback;
    typedef std::function<void(const std::string& job_id, const std::string& error)> ErrorCallback;
    
    void set_checkpoint_callback(CheckpointCallback callback);
    void set_error_callback(ErrorCallback callback);

private:
    std::string checkpoint_dir_;
    int auto_checkpoint_interval_seconds_;
    bool crash_recovery_enabled_;
    bool fast_checkpointing_enabled_;
    int compression_level_;
    
    std::mutex checkpoint_mutex_;
    std::atomic<bool> checkpointing_active_;
    
    CheckpointStats stats_;
    CheckpointCallback checkpoint_callback_;
    ErrorCallback error_callback_;
    
    // Background checkpoint thread
    std::unique_ptr<std::thread> auto_checkpoint_thread_;
    std::atomic<bool> auto_checkpoint_running_;
    
    // Internal methods
    std::string get_checkpoint_path(const std::string& job_id);
    std::string get_incremental_checkpoint_path(const std::string& job_id, int sequence);
    
    bool serialize_checkpoint_data(const CheckpointData& data, std::vector<uint8_t>& buffer);
    bool deserialize_checkpoint_data(const std::vector<uint8_t>& buffer, CheckpointData& data);
    
    bool write_checkpoint_file(const std::string& filepath, const std::vector<uint8_t>& data);
    bool read_checkpoint_file(const std::string& filepath, std::vector<uint8_t>& data);
    
    bool validate_checkpoint_data(const CheckpointData& data);
    std::string calculate_checkpoint_hash(const CheckpointData& data);
    
    void auto_checkpoint_worker();
    void cleanup_worker();
};

// Job Recovery System - High-level interface for job recovery
class JobRecoverySystem {
public:
    JobRecoverySystem(CheckpointManager* checkpoint_manager);
    
    enum class RecoveryStrategy {
        FULL_RESTART,           // Start job from beginning
        RESUME_FROM_CHECKPOINT, // Resume from last checkpoint
        SMART_RESUME,          // Analyze and choose best resume point
        PARTIAL_RESTART        // Restart problematic portions only
    };
    
    struct RecoveryPlan {
        RecoveryStrategy strategy;
        std::string reason;
        std::vector<std::string> required_actions;
        double estimated_recovery_time_seconds;
        double estimated_lost_progress_percentage;
        bool requires_user_intervention;
    };
    
    // Recovery analysis
    RecoveryPlan analyze_recovery_options(const std::string& job_id);
    
    bool execute_recovery_plan(
        const std::string& job_id,
        const RecoveryPlan& plan,
        CheckpointData& recovered_data
    );
    
    // Automatic recovery
    bool attempt_automatic_recovery(
        const std::string& job_id,
        CheckpointData& recovered_data
    );
    
    // Recovery validation
    bool validate_recovered_job(
        const std::string& job_id,
        const CheckpointData& data
    );
    
    // Recovery optimization
    std::vector<std::string> suggest_recovery_optimizations(
        const std::string& job_id
    );

private:
    CheckpointManager* checkpoint_manager_;
    
    bool can_resume_from_checkpoint(const CheckpointData& data);
    double calculate_resume_efficiency(const CheckpointData& data);
    bool detect_corruption_in_checkpoint(const CheckpointData& data);
};

// Real-time Progress Tracker
class ProgressTracker {
public:
    ProgressTracker();
    
    struct ProgressSnapshot {
        std::string job_id;
        double progress_percentage;
        uint64_t candidates_per_second;
        uint64_t total_candidates_tested;
        uint64_t hashes_cracked;
        std::chrono::duration<double> elapsed_time;
        std::chrono::duration<double> estimated_remaining;
        std::vector<double> device_utilization;
        double memory_usage_percentage;
        std::string current_phase; // "loading", "processing", "finishing"
    };
    
    void start_tracking(const std::string& job_id);
    void update_progress(const std::string& job_id, const ProgressSnapshot& snapshot);
    void stop_tracking(const std::string& job_id);
    
    ProgressSnapshot get_current_progress(const std::string& job_id);
    std::vector<ProgressSnapshot> get_progress_history(const std::string& job_id);
    
    // Performance prediction
    std::chrono::duration<double> predict_completion_time(const std::string& job_id);
    double predict_final_crack_count(const std::string& job_id);
    
    // Anomaly detection
    bool detect_performance_anomaly(const std::string& job_id);
    std::vector<std::string> suggest_performance_improvements(const std::string& job_id);

private:
    std::unordered_map<std::string, std::vector<ProgressSnapshot>> progress_history_;
    std::mutex progress_mutex_;
    
    double calculate_progress_velocity(const std::vector<ProgressSnapshot>& history);
    bool is_progress_stalled(const std::vector<ProgressSnapshot>& history);
};

// Utility functions for checkpoint management
namespace checkpoint_utils {
    // Checkpoint file operations
    bool backup_checkpoint(const std::string& checkpoint_path, const std::string& backup_path);
    bool restore_checkpoint_from_backup(const std::string& backup_path, const std::string& checkpoint_path);
    
    // Checkpoint validation
    bool verify_checkpoint_format(const std::string& checkpoint_path);
    std::vector<std::string> validate_checkpoint_consistency(const CheckpointData& data);
    
    // Storage optimization
    uint64_t estimate_checkpoint_size(const CheckpointData& data);
    double calculate_checkpoint_compression_ratio(const std::string& checkpoint_path);
    
    // Recovery analysis
    bool can_checkpoint_be_resumed(const CheckpointData& data);
    double estimate_recovery_success_probability(const CheckpointData& data);
    
    // Performance helpers
    std::chrono::duration<double> benchmark_checkpoint_creation(const CheckpointData& sample_data);
    std::chrono::duration<double> benchmark_checkpoint_restoration(const std::string& checkpoint_path);
    
    // Debugging and diagnostics
    std::string generate_checkpoint_report(const CheckpointData& data);
    bool export_checkpoint_for_analysis(const std::string& job_id, const std::string& output_path);
}