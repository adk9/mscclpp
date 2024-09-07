# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.25)
project(mscclpp-nccl-tests)

option(USE_CUDA "Use NVIDIA/CUDA." OFF)
option(USE_ROCM "Use AMD/ROCm." OFF)

if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(GPU_LIBS CUDA::cudart CUDA::cuda_driver)
    set(GPU_INCLUDE "${CUDAToolkit_INCLUDE_DIRS}")
elseif(USE_ROCM)
    find_package(hip REQUIRED)
    set(GPU_LIBS hip::device)
    set(GPU_INCLUDE "${hip_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Neither USE_CUDA nor USE_ROCM is set.")
endif()

find_package(MPI REQUIRED)
find_package(Threads REQUIRED)

file(GLOB_RECURSE COMMON_SOURCES CONFIGURE_DEPENDS src/common.cpp)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*_test.cpp)

set(NCCL_LIB "$ENV{NCCL_LIB}")
if(NOT NCCL_LIB)
    message("NCCL_LIB is not set, using default")
    set(NCCL_LIB ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/apps/nccl/libmscclpp_nccl.so)
endif()

set(NCCL_INCLUDE "$ENV{NCCL_INCLUDE}")
if(NOT NCCL_INCLUDE)
    message("NCCL_INCLUDE is not set, using default")
    set(NCCL_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
endif()

set(MSCCLPP_LIB "$ENV{MSCCLPP_LIB}")
if(NOT MSCCLPP_LIB)
    message("MSCCLPP_LIB is not set, using default")
    set(MSCCLPP_LIB ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/libmscclpp.so)
endif()

set(MSCCLPP_INCLUDE "$ENV{MSCCLPP_INCLUDE}")
if(NOT MSCCLPP_INCLUDE)
    message("MSCCLPP_INCLUDE is not set, using default")
    set(MSCCLPP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
endif()

include(CTest)
foreach(source IN ITEMS ${SOURCES})
    get_filename_component(test_name ${source} NAME_WE)
    add_executable(${test_name} ${source} ${COMMON_SOURCES})
    target_link_libraries(${test_name} ${NCCL_LIB} ${MSCCLPP_LIB} ${GPU_LIBS} Threads::Threads MPI::MPI_CXX)
    target_include_directories(${test_name} PRIVATE ${NCCL_INCLUDE} ${MSCCLPP_INCLUDE})
    add_test(NAME ${test_name}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND ${MPIEXEC_EXECUTABLE} --allow-run-as-root -np 8 ${test_name})
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 3600)
endforeach()
