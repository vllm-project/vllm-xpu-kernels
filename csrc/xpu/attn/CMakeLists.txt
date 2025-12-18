cmake_minimum_required(VERSION 3.18)

project(attn_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE ATTN_SOURCES "attn_interface.cpp")

add_library(attn_kernels STATIC ${ATTN_SOURCES})

target_include_directories(attn_kernels
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_SOURCE_DIR}
)

set(ATTN_LINK_LIBS)
set(VLLM_XPU_ENABLE_XE2 ON)
if(VLLM_XPU_ENABLE_XE2)
    add_subdirectory(xe_2)
    list(APPEND ATTN_LINK_LIBS attn_kernels_xe_2)
endif()

target_link_libraries(attn_kernels PUBLIC ${ATTN_LINK_LIBS} torch)
target_compile_options(attn_kernels PRIVATE -fPIC)