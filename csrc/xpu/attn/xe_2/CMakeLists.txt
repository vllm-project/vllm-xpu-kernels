cmake_minimum_required(VERSION 3.18)

set(ATTN_KERNEL_SRCS_GEN) # output
include("chunk_prefill_configure.cmake")
fmha_forward_configure(chunk_prefill_kernel_template)

# include("paged_decode_configure.cmake")
# paged_decode_configure(paged_decode_kernel_template)

# Exclude paged_decode sources when paged_decode is not configured
set(ATTN_KERNEL_SRCS_EXCLUDE)
list(APPEND ATTN_KERNEL_SRCS_EXCLUDE
  "${CMAKE_CURRENT_SOURCE_DIR}/paged_decode_xe2.cpp")

add_xe2_kernel_library(attn_kernels_xe_2 INCLUDE_CMAKE_SOURCE_DIR
  EXCLUDE_SOURCES ${ATTN_KERNEL_SRCS_EXCLUDE})
