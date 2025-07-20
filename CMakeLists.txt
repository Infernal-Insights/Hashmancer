cmake_minimum_required(VERSION 3.13)
project(darkling LANGUAGES C CXX)

option(ENABLE_CUDA "Enable CUDA backend" ON)
option(ENABLE_HIP "Enable HIP backend" ON)
option(ENABLE_OPENCL "Enable OpenCL backend" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/core)

file(GLOB CORE_SRC core/*.cpp core/utils/*.cpp)
file(GLOB LAUNCHER_SRC launcher/*.cpp)

set(DARKLING_SRCS ${CORE_SRC} ${LAUNCHER_SRC})

# CUDA Backend
if(ENABLE_CUDA)
  enable_language(CUDA)
  file(GLOB CUDA_SRC cuda_backend/*.cu)
  add_library(cuda_backend STATIC ${CUDA_SRC})
  target_compile_options(cuda_backend PRIVATE -O3 -Xcompiler -Wall)
  target_include_directories(cuda_backend PRIVATE ${CMAKE_SOURCE_DIR}/core)
  list(APPEND DARKLING_SRCS $<TARGET_OBJECTS:cuda_backend>)
endif()

# HIP Backend
if(ENABLE_HIP)
  find_package(HIP REQUIRED)
  file(GLOB HIP_SRC hip_backend/*.hip)
  hip_add_library(hip_backend STATIC ${HIP_SRC})
  target_include_directories(hip_backend PRIVATE ${CMAKE_SOURCE_DIR}/core)
  list(APPEND DARKLING_SRCS $<TARGET_OBJECTS:hip_backend>)
endif()

# OpenCL Backend
if(ENABLE_OPENCL)
  find_package(OpenCL REQUIRED)
  file(GLOB OPENCL_SRC intel_backend/*.cpp)
  add_library(opencl_backend STATIC ${OPENCL_SRC})
  target_include_directories(opencl_backend PRIVATE ${CMAKE_SOURCE_DIR}/core)
  target_link_libraries(opencl_backend PRIVATE OpenCL::OpenCL)
  list(APPEND DARKLING_SRCS $<TARGET_OBJECTS:opencl_backend>)
endif()

# Final Executable
add_executable(darkling_launcher ${DARKLING_SRCS})
target_compile_options(darkling_launcher PRIVATE -O3 -Wall)
