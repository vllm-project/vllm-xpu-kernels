cmake_minimum_required(VERSION 3.18)

set(ATTN_KERNEL_SRCS_GEN) # output
include("chunk_prefill_configure.cmake")
fmha_forward_configure(chunk_prefill_kernel_template)

include("paged_decode_configure.cmake")
paged_decode_configure(paged_decode_kernel_template)

add_xe2_kernel_library(attn_kernels_xe_2 INCLUDE_CMAKE_SOURCE_DIR)
# Fast e4m3 -> half reorder in SYCL-TLA (no NaN propagation, output scaled by
# 2^-8); the FMHA mainloops fold the 256 into the softmax and V scales. KV
# caches never hold NaN. No-op with SYCL-TLA versions that lack the option.
target_compile_definitions(attn_kernels_xe_2 PRIVATE CUTE_XE_E4M3_HALF_UNSCALED)
