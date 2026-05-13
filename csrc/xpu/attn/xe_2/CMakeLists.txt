cmake_minimum_required(VERSION 3.18)

set(ATTN_KERNEL_SRCS_GEN) # output
include("chunk_prefill_configure.cmake")
fmha_forward_configure(chunk_prefill_kernel_template)

include("paged_decode_configure.cmake")
paged_decode_configure(paged_decode_kernel_template)

add_xe2_kernel_library(attn_kernels_xe_2 INCLUDE_CMAKE_SOURCE_DIR)

# Shared PCH for the ~3.6k generated kernel TUs. The chunk_prefill +
# paged_decode kernel templates share almost their entire transitive
# include set (cute, CUTLASS-SYCL collective stack, sycl runtime); a
# single PCH collapses that re-parse cost into one drv. See
# attn_pch.hpp for the umbrella content. cmake handles the
# `-Xclang -include-pch` plumbing automatically; downstream non-cmake
# consumers (Nix per-TU dyn-drv builds) re-extract the PCH command
# from compile_commands.json and realise it as its own derivation.
target_precompile_headers(attn_kernels_xe_2 PRIVATE attn_pch.hpp)
