cmake_minimum_required(VERSION 3.18)
project(darkling LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Enable aggressive CUDA optimizations
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_90,code=compute_90")

# PTX optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -maxrregcount=64 -Xptxas -v,-warn-double-usage,-warn-spills")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DDARK_ENABLE_PTX_RULES=1")

# Custom command to compile PTX rules
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/ptx_rules.o
  COMMAND nvcc -ptx -O3 -arch=sm_80 
          ${CMAKE_SOURCE_DIR}/rules/ptx/rule_PREFIX_1.ptx 
          ${CMAKE_SOURCE_DIR}/rules/ptx/rule_SUFFIX_D4.ptx
          ${CMAKE_SOURCE_DIR}/rules/ptx/rule_VECTORIZED_COPY.ptx
          ${CMAKE_SOURCE_DIR}/rules/ptx/rule_CASE_TOGGLE_SIMD.ptx
          ${CMAKE_SOURCE_DIR}/rules/ptx/rule_LEET_LOOKUP_OPTIMIZED.ptx
          -o ${CMAKE_BINARY_DIR}/compiled_rules.ptx
  COMMAND ${CMAKE_CUDA_COMPILER} -c ${CMAKE_BINARY_DIR}/compiled_rules.ptx -o ${CMAKE_BINARY_DIR}/ptx_rules.o
  DEPENDS rules/ptx/rule_PREFIX_1.ptx 
          rules/ptx/rule_SUFFIX_D4.ptx
          rules/ptx/rule_VECTORIZED_COPY.ptx
          rules/ptx/rule_CASE_TOGGLE_SIMD.ptx
          rules/ptx/rule_LEET_LOOKUP_OPTIMIZED.ptx
  COMMENT "Compiling optimized PTX rules"
)

add_library(darkling
  src/dict_loader.cpp
  src/pinned_ringbuffer.cpp
  src/device_queue.cu
  src/dict_kernel_dispatch.cu
  src/rule_table.cu
  src/rule_dispatch.cu
  src/leet_lut.cu
  src/mask_engine.cu
  src/hybrid_attack_engine.cu
  src/smart_rule_engine.cu
  src/wordlist_optimizer.cu
  src/markov_chain_engine.cu
  src/rule_manager.cu
  src/rule_upload_handler.cpp
  src/hashcat_rule_parser.cpp
  ${CMAKE_BINARY_DIR}/ptx_rules.o
)

target_include_directories(darkling PUBLIC include)

# PTX Rule Compiler
add_executable(ptx_rule_compiler src/ptx_rule_compiler.cpp)
target_include_directories(ptx_rule_compiler PRIVATE include)

# Generate PTX rules from Best64 rule set
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/best64_rules.ptx
         ${CMAKE_BINARY_DIR}/best64_functions.h
  COMMAND ${CMAKE_BINARY_DIR}/ptx_rule_compiler ${CMAKE_BINARY_DIR}
  DEPENDS ptx_rule_compiler rules/best64.rule
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Generating PTX functions from Best64 rules"
)

add_custom_target(generate_ptx_rules 
  DEPENDS ${CMAKE_BINARY_DIR}/best64_rules.ptx 
          ${CMAKE_BINARY_DIR}/best64_functions.h
)

add_dependencies(darkling generate_ptx_rules)

# Main executables
add_executable(main src/main.cu)
target_link_libraries(main PRIVATE darkling)

add_executable(main_demo src/main_demo.cu)
target_link_libraries(main_demo PRIVATE darkling)

enable_testing()
add_executable(test_loader tests/test_loader.cpp)
add_executable(test_rules tests/test_rules.cpp)
add_executable(test_kernel tests/test_kernel.cu)
add_executable(test_rule_manager tests/test_rule_manager.cpp)
add_executable(test_rule_upload tests/test_rule_upload.cpp)

target_link_libraries(test_loader PRIVATE darkling)
target_link_libraries(test_rules PRIVATE darkling)
target_link_libraries(test_kernel PRIVATE darkling)
target_link_libraries(test_rule_manager PRIVATE darkling)
target_link_libraries(test_rule_upload PRIVATE darkling)

add_test(NAME test_loader COMMAND test_loader)
add_test(NAME test_rules COMMAND test_rules)
add_test(NAME test_kernel COMMAND test_kernel)
add_test(NAME test_rule_manager COMMAND test_rule_manager)
add_test(NAME test_rule_upload COMMAND test_rule_upload)
