cmake_minimum_required(VERSION 3.16)
project(llama_infer CXX CUDA)
include(cmake/cuda.cmake)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED  ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)
find_package(GTest REQUIRED)

aux_source_directory(inferer/source/tensor/ DIR_TENSOR)
aux_source_directory(inferer/source/base/ DIR_BASE)
aux_source_directory(inferer/source/op/ DIR_OP)
aux_source_directory(inferer/source/op/kernel/cpu DIR_KERNEL_CPU)
aux_source_directory(inferer/source/op/kernel/cuda DIR_KERNEL_CUDA)
aux_source_directory(inferer/source/op/kernel/ DIR_KERNEL)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(llama SHARED ${DIR_BASE} ${DIR_TENSOR} ${DIR_OP} ${DIR_KERNEL_CPU} ${DIR_KERNEL_CUDA} ${DIR_KERNEL})
target_link_libraries(llama cublas sentencepiece glog::glog gtest gtest_main pthread cudart ${ARMADILLO_LIBRARIES})
target_link_directories(llama PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

target_include_directories(llama PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${PROJECT_SOURCE_DIR}/inferer/include)
target_include_directories(llama PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(llama PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(llama_infer main.cpp)
target_link_directories(llama_infer PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(llama_infer llama)


add_subdirectory(test)
