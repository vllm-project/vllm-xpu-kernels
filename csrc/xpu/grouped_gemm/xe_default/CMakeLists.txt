cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE GROUPED_GEMM_SOURCES "*.cpp")

add_library(grouped_gemm_xe_default STATIC ${GROUPED_GEMM_SOURCES})

target_include_directories(grouped_gemm_xe_default
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_SOURCE_DIR}
)


target_compile_options(grouped_gemm_xe_default PRIVATE ${SYCL_TLA_KERNELS_COMPILE_FLAGS} -fPIC)
target_compile_definitions(grouped_gemm_xe_default PRIVATE -DVLLM_XPU_ENABLE_XE2)
target_include_directories(grouped_gemm_xe_default PRIVATE ${SYCL_TLA_INCLUDE_DIRS})

target_link_libraries(grouped_gemm_xe_default PRIVATE torch)
target_link_libraries(grouped_gemm_xe_default PRIVATE ${TORCH_LIBRARIES})

# FIXME(Kunshang): for default path, we'd better use AOT.
set(GROUPED_GEMM_GPU_LINK_FLAGS ${SYCL_DEVICE_LINK_FLAGS})

list(APPEND GROUPED_GEMM_GPU_LINK_FLAGS -Xsycl-target-backend=spir64_gen "-device ${AOT_DEVICES} -internal_options -cl-intel-256-GRF-per-thread")

target_link_options(grouped_gemm_xe_default PRIVATE ${GROUPED_GEMM_GPU_LINK_FLAGS})
