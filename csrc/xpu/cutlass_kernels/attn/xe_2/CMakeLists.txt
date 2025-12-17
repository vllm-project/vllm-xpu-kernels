cmake_minimum_required(VERSION 3.18)

project(attn_kernels LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE ATTN_SOURCES "*.cpp")

add_library(attn_xe_2 STATIC ${ATTN_SOURCES})

target_include_directories(attn_xe_2
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

message(STATUS "kunshang: FA2_VLLM_INCLUDE_DIRS: ${FA2_VLLM_INCLUDE_DIRS}")
target_compile_options(attn_xe_2 PRIVATE ${FA2_KERNELS_COMPILE_FLAGS} -fPIC)
target_include_directories(attn_xe_2 PRIVATE ${FA2_VLLM_INCLUDE_DIRS})

target_link_libraries(attn_xe_2 PRIVATE torch)
target_link_libraries(attn_xe_2 PRIVATE ${TORCH_LIBRARIES})

target_link_options(attn_xe_2 PRIVATE ${VLLM_GPU_LINK_FLAGS})


