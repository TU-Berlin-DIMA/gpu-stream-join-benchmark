cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(
        StreamJoinBenchmark
        VERSION 0.1
        DESCRIPTION "A Benchmarking Framework for Stream Join Algorithms"
        LANGUAGES CUDA CXX)

# Add Macros
include(cmake/macros.cmake)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    message("-- Debug mode - ON")
    #    ADD_DEFINITIONS(-DINFINITY_DEBUG_ON=1)
    ADD_DEFINITIONS(-DINFINITY_ASSERT_ON=1)
    ADD_DEFINITIONS(-DLOG_LEVEL=3)
    set(CMAKE_CXX_FLAGS "-g -pthread -O0 -march=native")
ELSE()
    set(CMAKE_CXX_FLAGS "-g -pthread -O3 -march=native")

ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

# CUDA SETUP
find_package(CUDAToolkit REQUIRED)
if(NOT ${CUDAToolkit_FOUND})
    message(FATAL_ERROR "Unable to find CUDAToolkit")
endif ()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(LIBRARIES ${LIBRARIES} CUDA::cudart CUDA::curand CUDA::nvml)
set(CMAKE_CUDA_FLAGS "-lineinfo --expt-relaxed-constexpr")
message("-- CUDA dependencies OK")

# TBB and YAML-CPP dependency
include(vcpkg/scripts/buildsystems/vcpkg.cmake)
find_package(TBB CONFIG REQUIRED)
set(LIBRARIES ${LIBRARIES} TBB::tbb)

find_package(yaml-cpp CONFIG REQUIRED)
set(LIBRARIES ${LIBRARIES} yaml-cpp)

# The compiled library code is here
add_subdirectory(src)
# The executable code is here
add_subdirectory(executable)
