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

add_subdirectory(xe_2)
set(ATTN_LINK_LIBS attn_xe_2)

target_link_libraries(attn_kernels PUBLIC ${ATTN_LINK_LIBS} torch)
target_compile_options(attn_kernels PRIVATE -fPIC)