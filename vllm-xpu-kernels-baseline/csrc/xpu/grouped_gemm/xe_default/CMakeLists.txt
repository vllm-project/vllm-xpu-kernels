cmake_minimum_required(VERSION 3.18)

# FIXME(Kunshang): for default path, we'd better use AOT.
add_xe_default_kernel_library(grouped_gemm_xe_default INCLUDE_CMAKE_SOURCE_DIR)
