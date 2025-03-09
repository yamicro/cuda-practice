cmake_minimum_required(VERSION 3.16)
project(llama_infer)
set(CMAKE_CXX_STANDARD 20)

find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)
find_package(GTest REQUIRED)

aux_source_directory(inferer/source/base/ DIR_BASE)



set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(llama SHARED ${DIR_BASE})
target_link_libraries(llama sentencepiece glog::glog gtest gtest_main pthread)
target_link_libraries(llama ${ARMADILLO_LIBRARIES})
target_include_directories(llama PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${PROJECT_SOURCE_DIR}/inferer/include)
target_include_directories(llama PUBLIC ${Armadillo_INCLUDE_DIR})

add_executable(llama_infer main.cpp)
target_link_directories(llama_infer PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(llama_infer llama)
add_subdirectory(test)
