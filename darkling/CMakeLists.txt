cmake_minimum_required(VERSION 3.18)
project(darkling LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_library(darkling
  src/dict_loader.cpp
  src/pinned_ringbuffer.cpp
  src/device_queue.cu
  src/dict_kernel_dispatch.cu
  src/rule_table.cu
  src/rule_dispatch.cu
  src/leet_lut.cu
)

target_include_directories(darkling PUBLIC include)

add_executable(main src/main.cu)
target_link_libraries(main PRIVATE darkling)

enable_testing()
add_executable(test_loader tests/test_loader.cpp)
add_executable(test_rules tests/test_rules.cpp)
add_executable(test_kernel tests/test_kernel.cu)

target_link_libraries(test_loader PRIVATE darkling)
target_link_libraries(test_rules PRIVATE darkling)
target_link_libraries(test_kernel PRIVATE darkling)

add_test(NAME test_loader COMMAND test_loader)
add_test(NAME test_rules COMMAND test_rules)
add_test(NAME test_kernel COMMAND test_kernel)
