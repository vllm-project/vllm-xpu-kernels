cmake_minimum_required(VERSION 3.18)

project(attn_kernels LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE ATTN_SOURCES "*.cpp")

add_library(attn_kernels_xe_2 STATIC ${ATTN_SOURCES})

target_include_directories(attn_kernels_xe_2
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

message(STATUS "kunshang: FA2_VLLM_INCLUDE_DIRS: ${FA2_VLLM_INCLUDE_DIRS}")
target_compile_options(attn_kernels_xe_2 PRIVATE ${FA2_KERNELS_COMPILE_FLAGS} -fPIC)
target_compile_definitions(attn_kernels_xe_2 PRIVATE -DVLLM_XPU_ENABLE_XE2)
target_include_directories(attn_kernels_xe_2 PRIVATE ${FA2_VLLM_INCLUDE_DIRS})

target_link_libraries(attn_kernels_xe_2 PRIVATE torch)
target_link_libraries(attn_kernels_xe_2 PRIVATE ${TORCH_LIBRARIES})

set(XE2_AOT_DEVICES "pvc,bmg-g21-a0")
set(FA2_GPU_LINK_FLAGS ${SYCL_DEVICE_LINK_FLAGS})

list(APPEND FA2_GPU_LINK_FLAGS -Xsycl-target-backend=spir64_gen "-device ${XE2_AOT_DEVICES} -internal_options -cl-intel-256-GRF-per-thread")

target_link_options(attn_kernels_xe_2 PRIVATE ${FA2_GPU_LINK_FLAGS})


