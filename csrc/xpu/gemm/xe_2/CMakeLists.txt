cmake_minimum_required(VERSION 3.18)

set(GEMM_KERNEL_SRCS_GEN)
include("gemm_xe2_configure.cmake")
gemm_xe2_configure(gemm_xe2_kernel_template)

add_xe2_kernel_library(gemm_xe_2 INCLUDE_CMAKE_SOURCE_DIR)
