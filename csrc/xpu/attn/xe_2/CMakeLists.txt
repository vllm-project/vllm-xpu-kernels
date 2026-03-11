cmake_minimum_required(VERSION 3.18)

set(ATTN_KERNEL_SRCS_GEN) # output
include("chunk_prefill_configure.cmake")
fmha_forward_configure(chunk_prefill_kernel_template)

include("paged_decode_configure.cmake")
paged_decode_configure(paged_decode_kernel_template)
set(ATTN_DECODE_SRCS_GEN ${GEN_KERNEL_SRCS})

# Split generated chunk prefill instantiations to reduce peak link pressure.
set(ATTN_PREFILL_SRCS_FP8Q ${ATTN_KERNEL_SRCS_GEN_FP8Q})
set(ATTN_PREFILL_SRCS_NON_FP8Q ${ATTN_KERNEL_SRCS_GEN_NON_FP8Q})
set(ATTN_DECODE_SRCS ${ATTN_DECODE_SRCS_GEN})

# Keep runtime entry points in dedicated prefill/decode binaries.
list(APPEND ATTN_PREFILL_SRCS_NON_FP8Q ${CMAKE_CURRENT_SOURCE_DIR}/fmha_xe2.cpp)
list(APPEND ATTN_DECODE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/paged_decode_xe2.cpp)

add_xe2_kernel_library(attn_prefill_kernels_xe_2 INCLUDE_CMAKE_SOURCE_DIR
                       SOURCES ${ATTN_PREFILL_SRCS_NON_FP8Q})

if(ATTN_PREFILL_SRCS_FP8Q)
  add_xe2_kernel_library(
    attn_prefill_kernels_xe_2_fp8q INCLUDE_CMAKE_SOURCE_DIR SOURCES
    ${ATTN_PREFILL_SRCS_FP8Q})
  set(ATTN_HAS_FP8Q_LIB
      ON
      PARENT_SCOPE)
else()
  set(ATTN_HAS_FP8Q_LIB
      OFF
      PARENT_SCOPE)
endif()

add_xe2_kernel_library(attn_decode_kernels_xe_2 INCLUDE_CMAKE_SOURCE_DIR
                       SOURCES ${ATTN_DECODE_SRCS})
