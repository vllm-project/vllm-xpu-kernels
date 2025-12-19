cmake_minimum_required(VERSION 3.26)

project(grouped_gemm LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE GROUPED_GEMM_SOURCES "grouped_gemm_interface.cpp")

add_library(grouped_gemm STATIC ${GROUPED_GEMM_SOURCES})

target_include_directories(grouped_gemm
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_SOURCE_DIR}
)

set(GROUPED_GEMM_LINK_LIBS)
set(VLLM_XPU_ENABLE_XE2 ON)
if(VLLM_XPU_ENABLE_XE2)
    add_subdirectory(xe_2)
    list(APPEND GROUPED_GEMM_LINK_LIBS grouped_gemm_xe_2)
endif()
set(VLLM_XPU_ENABLE_XE_DEFAULT ON)
if(VLLM_XPU_ENABLE_XE_DEFAULT)
    add_subdirectory(xe_default)
    list(APPEND GROUPED_GEMM_LINK_LIBS grouped_gemm_xe_default)
endif()

target_link_libraries(grouped_gemm PUBLIC ${GROUPED_GEMM_LINK_LIBS} torch)
target_compile_options(grouped_gemm PRIVATE -fPIC)