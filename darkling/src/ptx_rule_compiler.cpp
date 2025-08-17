#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include "darkling_rule_manager.h"

// PTX Rule Compiler - Converts hashcat rules to optimized PTX assembly
// This generates high-performance PTX functions for the most common rules

class PTXRuleCompiler {
private:
    std::unordered_map<std::string, std::string> ptx_templates;
    std::vector<std::string> generated_functions;
    uint32_t function_counter;
    
public:
    PTXRuleCompiler() : function_counter(0) {
        initialize_templates();
    }
    
    void initialize_templates() {
        // PTX template for basic character operations
        ptx_templates["header"] = R"(
.version 8.0
.target sm_80
.address_size 64

)";
        
        // Template for no-operation (pass through)
        ptx_templates["noop"] = R"(
.visible .func rule_noop(
    .param .u64 output_ptr,
    .param .u64 input_ptr,
    .param .u32 input_len,
    .param .u64 params_ptr,
    .param .u32 variant_idx)
{
    .reg .u64 %rd<10>;
    .reg .u32 %r<10>;
    .reg .u8  %b<5>;
    
    ld.param.u64 %rd1, [output_ptr];
    ld.param.u64 %rd2, [input_ptr];
    ld.param.u32 %r1, [input_len];
    
    // Simple copy
    mov.u32 %r2, 0;
copy_loop:
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra copy_done;
    
    add.u64 %rd3, %rd2, %r2;
    add.u64 %rd4, %rd1, %r2;
    ld.u8 %b1, [%rd3];
    st.u8 [%rd4], %b1;
    
    add.u32 %r2, %r2, 1;
    bra copy_loop;
    
copy_done:
    ret;
}
)";

        // Template for lowercase transformation
        ptx_templates["lowercase"] = R"(
.visible .func rule_lowercase(
    .param .u64 output_ptr,
    .param .u64 input_ptr,
    .param .u32 input_len,
    .param .u64 params_ptr,
    .param .u32 variant_idx)
{
    .reg .u64 %rd<10>;
    .reg .u32 %r<10>;
    .reg .u8  %b<5>;
    .reg .pred %p<5>;
    
    ld.param.u64 %rd1, [output_ptr];
    ld.param.u64 %rd2, [input_ptr];
    ld.param.u32 %r1, [input_len];
    
    mov.u32 %r2, 0;
transform_loop:
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra transform_done;
    
    add.u64 %rd3, %rd2, %r2;
    add.u64 %rd4, %rd1, %r2;
    ld.u8 %b1, [%rd3];
    
    // Convert to lowercase if uppercase
    setp.ge.u32 %p2, %b1, 65;    // >= 'A'
    setp.le.u32 %p3, %b1, 90;    // <= 'Z'
    and.pred %p4, %p2, %p3;      // Is uppercase?
    
    selp.u32 %r3, 32, 0, %p4;    // Add 32 if uppercase
    add.u32 %r4, %b1, %r3;
    cvt.u8.u32 %b2, %r4;
    st.u8 [%rd4], %b2;
    
    add.u32 %r2, %r2, 1;
    bra transform_loop;
    
transform_done:
    ret;
}
)";

        // Template for uppercase transformation
        ptx_templates["uppercase"] = R"(
.visible .func rule_uppercase(
    .param .u64 output_ptr,
    .param .u64 input_ptr,
    .param .u32 input_len,
    .param .u64 params_ptr,
    .param .u32 variant_idx)
{
    .reg .u64 %rd<10>;
    .reg .u32 %r<10>;
    .reg .u8  %b<5>;
    .reg .pred %p<5>;
    
    ld.param.u64 %rd1, [output_ptr];
    ld.param.u64 %rd2, [input_ptr];
    ld.param.u32 %r1, [input_len];
    
    mov.u32 %r2, 0;
transform_loop:
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra transform_done;
    
    add.u64 %rd3, %rd2, %r2;
    add.u64 %rd4, %rd1, %r2;
    ld.u8 %b1, [%rd3];
    
    // Convert to uppercase if lowercase
    setp.ge.u32 %p2, %b1, 97;    // >= 'a'
    setp.le.u32 %p3, %b1, 122;   // <= 'z'
    and.pred %p4, %p2, %p3;      // Is lowercase?
    
    selp.u32 %r3, -32, 0, %p4;   // Subtract 32 if lowercase
    add.u32 %r4, %b1, %r3;
    cvt.u8.u32 %b2, %r4;
    st.u8 [%rd4], %b2;
    
    add.u32 %r2, %r2, 1;
    bra transform_loop;
    
transform_done:
    ret;
}
)";

        // Template for character append
        ptx_templates["append_char"] = R"(
.visible .func rule_append_char(
    .param .u64 output_ptr,
    .param .u64 input_ptr,
    .param .u32 input_len,
    .param .u64 params_ptr,
    .param .u32 variant_idx)
{
    .reg .u64 %rd<10>;
    .reg .u32 %r<10>;
    .reg .u8  %b<5>;
    
    ld.param.u64 %rd1, [output_ptr];
    ld.param.u64 %rd2, [input_ptr];
    ld.param.u32 %r1, [input_len];
    ld.param.u64 %rd5, [params_ptr];
    
    // Copy original string
    mov.u32 %r2, 0;
copy_loop:
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra copy_done;
    
    add.u64 %rd3, %rd2, %r2;
    add.u64 %rd4, %rd1, %r2;
    ld.u8 %b1, [%rd3];
    st.u8 [%rd4], %b1;
    
    add.u32 %r2, %r2, 1;
    bra copy_loop;
    
copy_done:
    // Append character from params
    ld.u8 %b2, [%rd5];           // Load append character
    add.u64 %rd6, %rd1, %r1;     // output + input_len
    st.u8 [%rd6], %b2;           // Store append character
    
    ret;
}
)";

        // Template for character prepend
        ptx_templates["prepend_char"] = R"(
.visible .func rule_prepend_char(
    .param .u64 output_ptr,
    .param .u64 input_ptr,
    .param .u32 input_len,
    .param .u64 params_ptr,
    .param .u32 variant_idx)
{
    .reg .u64 %rd<10>;
    .reg .u32 %r<10>;
    .reg .u8  %b<5>;
    
    ld.param.u64 %rd1, [output_ptr];
    ld.param.u64 %rd2, [input_ptr];
    ld.param.u32 %r1, [input_len];
    ld.param.u64 %rd5, [params_ptr];
    
    // Store prepend character first
    ld.u8 %b1, [%rd5];
    st.u8 [%rd1], %b1;
    
    // Copy original string shifted by 1
    mov.u32 %r2, 0;
copy_loop:
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra copy_done;
    
    add.u64 %rd3, %rd2, %r2;     // input + i
    add.u64 %rd4, %rd1, %r2;     // output + i
    add.u64 %rd4, %rd4, 1;       // output + i + 1
    ld.u8 %b2, [%rd3];
    st.u8 [%rd4], %b2;
    
    add.u32 %r2, %r2, 1;
    bra copy_loop;
    
copy_done:
    ret;
}
)";

        // Template for reverse string
        ptx_templates["reverse"] = R"(
.visible .func rule_reverse(
    .param .u64 output_ptr,
    .param .u64 input_ptr,
    .param .u32 input_len,
    .param .u64 params_ptr,
    .param .u32 variant_idx)
{
    .reg .u64 %rd<10>;
    .reg .u32 %r<10>;
    .reg .u8  %b<5>;
    
    ld.param.u64 %rd1, [output_ptr];
    ld.param.u64 %rd2, [input_ptr];
    ld.param.u32 %r1, [input_len];
    
    mov.u32 %r2, 0;
reverse_loop:
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra reverse_done;
    
    // input[i] -> output[len-1-i]
    sub.u32 %r3, %r1, 1;         // len - 1
    sub.u32 %r3, %r3, %r2;       // len - 1 - i
    
    add.u64 %rd3, %rd2, %r2;     // input + i
    add.u64 %rd4, %rd1, %r3;     // output + (len-1-i)
    ld.u8 %b1, [%rd3];
    st.u8 [%rd4], %b1;
    
    add.u32 %r2, %r2, 1;
    bra reverse_loop;
    
reverse_done:
    ret;
}
)";

        // Template for duplicate (word + word)
        ptx_templates["duplicate"] = R"(
.visible .func rule_duplicate(
    .param .u64 output_ptr,
    .param .u64 input_ptr,
    .param .u32 input_len,
    .param .u64 params_ptr,
    .param .u32 variant_idx)
{
    .reg .u64 %rd<10>;
    .reg .u32 %r<10>;
    .reg .u8  %b<5>;
    
    ld.param.u64 %rd1, [output_ptr];
    ld.param.u64 %rd2, [input_ptr];
    ld.param.u32 %r1, [input_len];
    
    // Copy first instance
    mov.u32 %r2, 0;
copy1_loop:
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra copy1_done;
    
    add.u64 %rd3, %rd2, %r2;
    add.u64 %rd4, %rd1, %r2;
    ld.u8 %b1, [%rd3];
    st.u8 [%rd4], %b1;
    
    add.u32 %r2, %r2, 1;
    bra copy1_loop;
    
copy1_done:
    // Copy second instance
    mov.u32 %r3, 0;
copy2_loop:
    setp.ge.u32 %p2, %r3, %r1;
    @%p2 bra copy2_done;
    
    add.u64 %rd5, %rd2, %r3;     // input + i
    add.u64 %rd6, %rd1, %r1;     // output + input_len
    add.u64 %rd6, %rd6, %r3;     // output + input_len + i
    ld.u8 %b2, [%rd5];
    st.u8 [%rd6], %b2;
    
    add.u32 %r3, %r3, 1;
    bra copy2_loop;
    
copy2_done:
    ret;
}
)";
    }
    
    std::string compile_rule(const std::string& rule_string, const std::string& function_name) {
        if (rule_string == ":") {
            return generate_function(function_name, "noop", {});
        }
        else if (rule_string == "l") {
            return generate_function(function_name, "lowercase", {});
        }
        else if (rule_string == "u") {
            return generate_function(function_name, "uppercase", {});
        }
        else if (rule_string == "r") {
            return generate_function(function_name, "reverse", {});
        }
        else if (rule_string == "d") {
            return generate_function(function_name, "duplicate", {});
        }
        else if (rule_string.length() == 2 && rule_string[0] == '$') {
            // Append character
            return generate_append_function(function_name, rule_string[1]);
        }
        else if (rule_string.length() == 2 && rule_string[0] == '^') {
            // Prepend character
            return generate_prepend_function(function_name, rule_string[1]);
        }
        else if (rule_string.length() == 3 && rule_string[0] == 's') {
            // Character substitution
            return generate_substitute_function(function_name, rule_string[1], rule_string[2]);
        }
        else if (rule_string == "c") {
            return generate_capitalize_function(function_name);
        }
        else if (rule_string == "C") {
            return generate_invert_case_function(function_name);
        }
        else if (rule_string == "t") {
            return generate_toggle_case_function(function_name);
        }
        else if (rule_string == "[") {
            return generate_delete_first_function(function_name);
        }
        else if (rule_string == "]") {
            return generate_delete_last_function(function_name);
        }
        else {
            // Fallback to interpreted rule
            return "";
        }
    }
    
    std::string generate_function(const std::string& name, const std::string& template_name, 
                                 const std::vector<std::string>& replacements) {
        std::string ptx_code = ptx_templates[template_name];
        
        // Replace function name
        size_t pos = ptx_code.find("rule_" + template_name);
        if (pos != std::string::npos) {
            ptx_code.replace(pos, template_name.length() + 5, name);
        }
        
        return ptx_code;
    }
    
    std::string generate_append_function(const std::string& name, char append_char) {
        std::string ptx_code = ptx_templates["append_char"];
        
        // Replace function name
        size_t pos = ptx_code.find("rule_append_char");
        if (pos != std::string::npos) {
            ptx_code.replace(pos, 16, name);
        }
        
        // Replace the character loading with immediate value
        std::stringstream ss;
        ss << "    mov.u8 %b2, " << (int)append_char << ";           // Append '" << append_char << "'\n";
        ss << "    add.u64 %rd6, %rd1, %r1;     // output + input_len\n";
        ss << "    st.u8 [%rd6], %b2;           // Store append character";
        
        size_t param_pos = ptx_code.find("ld.u8 %b2, [%rd5];");
        if (param_pos != std::string::npos) {
            size_t end_pos = ptx_code.find("st.u8 [%rd6], %b2;", param_pos);
            end_pos = ptx_code.find('\n', end_pos) + 1;
            ptx_code.replace(param_pos, end_pos - param_pos, ss.str());
        }
        
        return ptx_code;
    }
    
    std::string generate_prepend_function(const std::string& name, char prepend_char) {
        std::string ptx_code = ptx_templates["prepend_char"];
        
        size_t pos = ptx_code.find("rule_prepend_char");
        if (pos != std::string::npos) {
            ptx_code.replace(pos, 17, name);
        }
        
        // Replace character loading
        std::stringstream ss;
        ss << "    mov.u8 %b1, " << (int)prepend_char << ";";
        
        pos = ptx_code.find("ld.u8 %b1, [%rd5];");
        if (pos != std::string::npos) {
            ptx_code.replace(pos, 19, ss.str());
        }
        
        return ptx_code;
    }
    
    std::string generate_substitute_function(const std::string& name, char from_char, char to_char) {
        std::stringstream ss;
        ss << R"(
.visible .func )" << name << R"((
    .param .u64 output_ptr,
    .param .u64 input_ptr,
    .param .u32 input_len,
    .param .u64 params_ptr,
    .param .u32 variant_idx)
{
    .reg .u64 %rd<10>;
    .reg .u32 %r<10>;
    .reg .u8  %b<5>;
    .reg .pred %p<5>;
    
    ld.param.u64 %rd1, [output_ptr];
    ld.param.u64 %rd2, [input_ptr];
    ld.param.u32 %r1, [input_len];
    
    mov.u32 %r2, 0;
substitute_loop:
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra substitute_done;
    
    add.u64 %rd3, %rd2, %r2;
    add.u64 %rd4, %rd1, %r2;
    ld.u8 %b1, [%rd3];
    
    // Check if character matches
    setp.eq.u32 %p2, %b1, )" << (int)from_char << R"(;
    selp.u32 %r3, )" << (int)to_char << R"(, %b1, %p2;
    cvt.u8.u32 %b2, %r3;
    st.u8 [%rd4], %b2;
    
    add.u32 %r2, %r2, 1;
    bra substitute_loop;
    
substitute_done:
    ret;
}
)";
        return ss.str();
    }
    
    std::string generate_capitalize_function(const std::string& name) {
        std::stringstream ss;
        ss << R"(
.visible .func )" << name << R"((
    .param .u64 output_ptr,
    .param .u64 input_ptr,
    .param .u32 input_len,
    .param .u64 params_ptr,
    .param .u32 variant_idx)
{
    .reg .u64 %rd<10>;
    .reg .u32 %r<10>;
    .reg .u8  %b<5>;
    .reg .pred %p<5>;
    
    ld.param.u64 %rd1, [output_ptr];
    ld.param.u64 %rd2, [input_ptr];
    ld.param.u32 %r1, [input_len];
    
    // Handle first character (capitalize)
    setp.eq.u32 %p1, %r1, 0;
    @%p1 bra cap_done;
    
    ld.u8 %b1, [%rd2];
    setp.ge.u32 %p2, %b1, 97;    // >= 'a'
    setp.le.u32 %p3, %b1, 122;   // <= 'z'
    and.pred %p4, %p2, %p3;      // Is lowercase?
    selp.u32 %r2, -32, 0, %p4;   // Convert to uppercase
    add.u32 %r3, %b1, %r2;
    cvt.u8.u32 %b2, %r3;
    st.u8 [%rd1], %b2;
    
    // Copy remaining characters as lowercase
    mov.u32 %r4, 1;
cap_loop:
    setp.ge.u32 %p5, %r4, %r1;
    @%p5 bra cap_done;
    
    add.u64 %rd3, %rd2, %r4;
    add.u64 %rd4, %rd1, %r4;
    ld.u8 %b3, [%rd3];
    
    setp.ge.u32 %p6, %b3, 65;    // >= 'A'
    setp.le.u32 %p7, %b3, 90;    // <= 'Z'
    and.pred %p8, %p6, %p7;      // Is uppercase?
    selp.u32 %r5, 32, 0, %p8;    // Convert to lowercase
    add.u32 %r6, %b3, %r5;
    cvt.u8.u32 %b4, %r6;
    st.u8 [%rd4], %b4;
    
    add.u32 %r4, %r4, 1;
    bra cap_loop;
    
cap_done:
    ret;
}
)";
        return ss.str();
    }
    
    std::string generate_invert_case_function(const std::string& name) {
        std::stringstream ss;
        ss << R"(
.visible .func )" << name << R"((
    .param .u64 output_ptr,
    .param .u64 input_ptr,
    .param .u32 input_len,
    .param .u64 params_ptr,
    .param .u32 variant_idx)
{
    .reg .u64 %rd<10>;
    .reg .u32 %r<10>;
    .reg .u8  %b<5>;
    .reg .pred %p<10>;
    
    ld.param.u64 %rd1, [output_ptr];
    ld.param.u64 %rd2, [input_ptr];
    ld.param.u32 %r1, [input_len];
    
    mov.u32 %r2, 0;
invert_loop:
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra invert_done;
    
    add.u64 %rd3, %rd2, %r2;
    add.u64 %rd4, %rd1, %r2;
    ld.u8 %b1, [%rd3];
    
    // Check if uppercase
    setp.ge.u32 %p2, %b1, 65;    // >= 'A'
    setp.le.u32 %p3, %b1, 90;    // <= 'Z'
    and.pred %p4, %p2, %p3;      // Is uppercase?
    
    // Check if lowercase
    setp.ge.u32 %p5, %b1, 97;    // >= 'a'
    setp.le.u32 %p6, %b1, 122;   // <= 'z'
    and.pred %p7, %p5, %p6;      // Is lowercase?
    
    // Convert: uppercase->lowercase, lowercase->uppercase
    selp.u32 %r3, 32, 0, %p4;    // +32 if uppercase
    selp.u32 %r4, -32, 0, %p7;   // -32 if lowercase
    add.u32 %r5, %r3, %r4;       // Combined offset
    add.u32 %r6, %b1, %r5;       // Apply transformation
    
    cvt.u8.u32 %b2, %r6;
    st.u8 [%rd4], %b2;
    
    add.u32 %r2, %r2, 1;
    bra invert_loop;
    
invert_done:
    ret;
}
)";
        return ss.str();
    }
    
    std::string generate_toggle_case_function(const std::string& name) {
        // Similar to invert_case but toggles every character
        return generate_invert_case_function(name);
    }
    
    std::string generate_delete_first_function(const std::string& name) {
        std::stringstream ss;
        ss << R"(
.visible .func )" << name << R"((
    .param .u64 output_ptr,
    .param .u64 input_ptr,
    .param .u32 input_len,
    .param .u64 params_ptr,
    .param .u32 variant_idx)
{
    .reg .u64 %rd<10>;
    .reg .u32 %r<10>;
    .reg .u8  %b<5>;
    
    ld.param.u64 %rd1, [output_ptr];
    ld.param.u64 %rd2, [input_ptr];
    ld.param.u32 %r1, [input_len];
    
    // Skip first character, copy rest
    setp.le.u32 %p1, %r1, 1;
    @%p1 bra delete_done;
    
    mov.u32 %r2, 1;              // Start from second character
delete_loop:
    setp.ge.u32 %p2, %r2, %r1;
    @%p2 bra delete_done;
    
    add.u64 %rd3, %rd2, %r2;     // input + i
    sub.u32 %r3, %r2, 1;         // i - 1
    add.u64 %rd4, %rd1, %r3;     // output + (i-1)
    ld.u8 %b1, [%rd3];
    st.u8 [%rd4], %b1;
    
    add.u32 %r2, %r2, 1;
    bra delete_loop;
    
delete_done:
    ret;
}
)";
        return ss.str();
    }
    
    std::string generate_delete_last_function(const std::string& name) {
        std::stringstream ss;
        ss << R"(
.visible .func )" << name << R"((
    .param .u64 output_ptr,
    .param .u64 input_ptr,
    .param .u32 input_len,
    .param .u64 params_ptr,
    .param .u32 variant_idx)
{
    .reg .u64 %rd<10>;
    .reg .u32 %r<10>;
    .reg .u8  %b<5>;
    
    ld.param.u64 %rd1, [output_ptr];
    ld.param.u64 %rd2, [input_ptr];
    ld.param.u32 %r1, [input_len];
    
    // Copy all but last character
    setp.eq.u32 %p1, %r1, 0;
    @%p1 bra delete_done;
    
    sub.u32 %r5, %r1, 1;         // target_len = input_len - 1
    mov.u32 %r2, 0;
delete_loop:
    setp.ge.u32 %p2, %r2, %r5;
    @%p2 bra delete_done;
    
    add.u64 %rd3, %rd2, %r2;
    add.u64 %rd4, %rd1, %r2;
    ld.u8 %b1, [%rd3];
    st.u8 [%rd4], %b1;
    
    add.u32 %r2, %r2, 1;
    bra delete_loop;
    
delete_done:
    ret;
}
)";
        return ss.str();
    }
    
    bool compile_best64_rules(const std::string& output_path) {
        std::ifstream rule_file("rules/best64.rule");
        if (!rule_file.is_open()) {
            std::cerr << "Could not open best64.rule file" << std::endl;
            return false;
        }
        
        std::ofstream output_file(output_path);
        if (!output_file.is_open()) {
            std::cerr << "Could not create output file: " << output_path << std::endl;
            return false;
        }
        
        // Write PTX header
        output_file << ptx_templates["header"];
        
        std::string line;
        uint32_t rule_index = 0;
        
        while (std::getline(rule_file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#') continue;
            
            std::string function_name = "rule_best64_" + std::to_string(rule_index);
            std::string ptx_function = compile_rule(line, function_name);
            
            if (!ptx_function.empty()) {
                output_file << ptx_function << "\n";
                generated_functions.push_back(function_name);
            }
            
            rule_index++;
        }
        
        output_file.close();
        
        std::cout << "Generated " << generated_functions.size() << " PTX rule functions" << std::endl;
        return true;
    }
    
    void generate_function_table_header(const std::string& output_path) {
        std::ofstream header_file(output_path);
        if (!header_file.is_open()) return;
        
        header_file << "// Auto-generated PTX rule function declarations\n";
        header_file << "#pragma once\n\n";
        
        header_file << "extern \"C\" {\n";
        for (const auto& func_name : generated_functions) {
            header_file << "extern __device__ void " << func_name 
                       << "(uint8_t*, const uint8_t*, uint32_t, const DlRuleParams*, uint32_t);\n";
        }
        header_file << "}\n\n";
        
        header_file << "// PTX rule function table\n";
        header_file << "static const DlPTXRuleFunc ptx_best64_functions[] = {\n";
        for (const auto& func_name : generated_functions) {
            header_file << "    " << func_name << ",\n";
        }
        header_file << "};\n";
        
        header_file.close();
    }
};

// Command line tool for PTX rule compilation
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <output_directory>" << std::endl;
        return 1;
    }
    
    std::string output_dir = argv[1];
    PTXRuleCompiler compiler;
    
    if (!compiler.compile_best64_rules(output_dir + "/best64_rules.ptx")) {
        std::cerr << "Failed to compile Best64 rules" << std::endl;
        return 1;
    }
    
    compiler.generate_function_table_header(output_dir + "/best64_functions.h");
    
    std::cout << "PTX rule compilation completed successfully!" << std::endl;
    return 0;
}