cmake_minimum_required(VERSION 3.18)

include("chunk_prefill_configure.cmake")
fmha_forward_configure()

add_xe2_kernel_library(attn_kernels_xe_2 INCLUDE_CMAKE_SOURCE_DIR)


